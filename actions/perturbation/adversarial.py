import ast

from OpenAttack import attackers
from OpenAttack.victim import classifiers
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from OpenAttack.utils.visualizer import sent_len, levenshtein_visual
import numpy as np

from explained_models.ModelABC.DANetwork import DANetwork
from explained_models.Tokenizer.tokenizer import HFTokenizer
from actions.perturbation.attack.attack_eval import AttackEval


class AdversarialDataset(Dataset):
    def __init__(self, dataset_json):
        self.data = [dataset_json]

    def __len__(self):
        return len(self.data)
        # return 1

    def __getitem__(self, idx):
        return self.data[idx]


def adversarial_operation(conversation, parse_text, i, simulation, **kwargs):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    for item in parse_text:
        try:
            _id = int(item)
        except ValueError:
            pass

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    dataset_name = conversation.describe.get_dataset_name()

    if dataset_name == 'boolq':
        num_col = 2
    else:
        num_col = 1

    f_names = list(conversation.temp_dataset.contents['X'].columns)
    texts = conversation.temp_dataset.contents['X']
    label = conversation.temp_dataset.contents['y'][_id]
    filtered_text = ''

    dataset_json = {}

    for f in f_names[:num_col]:
        filtered_text += texts[f][_id]
        if dataset_name == 'boolq':
            filtered_text += " "

    dataset_json['x'] = filtered_text
    dataset_json["y"] = label

    dataset = AdversarialDataset(dataset_json)

    if dataset_name == 'boolq':
        model = AutoModelForSequenceClassification.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")
        tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")
        victim = classifiers.TransformersClassifier(model, tokenizer, model.base_model.embeddings.word_embeddings)
    elif dataset_name == 'daily_dialog':
        model = DANetwork()
        tokenizer = HFTokenizer('bert-base-uncased', mode='bert').tokenizer
        victim = classifiers.TransformersClassifier(model.bert, tokenizer, model.bert.base_model.embeddings.word_embeddings)
    elif dataset_name == 'olid':
        model = AutoModelForSequenceClassification.from_pretrained("sinhala-nlp/mbert-olid-en")
        tokenizer = AutoTokenizer.from_pretrained("sinhala-nlp/mbert-olid-en")
        victim = classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported!")

    attacker = attackers.PWWSAttacker()

    # prepare for attacking
    attack_eval = AttackEval(attacker, victim)

    # launch attacks and print attack results
    d = attack_eval.eval(dataset, visualize=False)

    return_s = ""

    x_orig = d["x_orig"]
    x_adv = d["x_adv"]
    y_orig = ast.literal_eval(d["y_orig"])
    y_adv = ast.literal_eval(d["y_adv"])
    token_orig = tokenizer.tokenize(x_orig)
    if x_adv:
        token_adv = tokenizer.tokenize(x_adv)
    else:
        token_adv = token_orig
        return_s += "Attack was unsuccessful.<br>"

    pairs = levenshtein_visual(token_orig, token_adv)

    curr1 = ""
    curr2 = ""
    max_len = 200
    length = 0
    ret = []
    for tokenA, tokenB in pairs:
        assert sent_len(tokenA) == sent_len(tokenB)
        if length + sent_len(tokenA) + 1 > max_len:
            ret.append(curr1 + " " * (max_len - length))
            ret.append(curr2 + " " * (max_len - length))
            ret.append(" " * max_len)
            length = sent_len(tokenA) + 1
            if tokenA.lower() == tokenB.lower():
                curr1 = tokenA + " "
                curr2 = tokenB + " "
            else:
                curr1 = "<span style='color:red;font-weight:bold'>" + tokenA + "</span>" + " "
                curr2 = "<span style='color:green;font-weight:bold'>" + tokenB + "</span>" + " "
        else:
            length += 1 + sent_len(tokenA)
            if tokenA.lower() == tokenB.lower():
                curr1 += tokenA + " "
                curr2 += tokenB + " "
            else:
                curr1 += "<span style='color:red;font-weight:bold'>" + tokenA + "</span>" + " "
                curr2 += "<span style='color:green;font-weight:bold'>" + tokenB + "</span>" + " "
    if length > 0:
        ret.append(curr1 + " " * (max_len - length))
        ret.append(curr2 + " " * (max_len - length))
        ret.append(" " * max_len)

    idx_orig = np.argmax(y_orig)
    idx_adv = np.argmax(y_adv)
    prob_orig = round(y_orig[idx_orig] * 100, conversation.rounding_precision)
    prob_adv = round(y_adv[idx_adv] * 100, conversation.rounding_precision)
    if not simulation:
        return_s += f"<span style='color:green;font-weight:bold'>Label {conversation.class_names[idx_orig]} ({prob_orig}%) --> {conversation.class_names[idx_adv]} ({prob_adv}%)</span> <br><br>"

    for i in range(0, len(ret)):
        return_s += ret[i]
        return_s += "<br>"

        if i % 2 != 0:
            return_s += '<br>'

    return return_s, 1
