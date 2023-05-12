import json

from OpenAttack import attackers, AttackEval
from OpenAttack.victim import classifiers
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import OpenAttack as oa


class CustomInputDataset(Dataset):
    def __init__(self, dataset_json, dataset_name):
        self.data = []

        if dataset_name == "boolq":
            self.data.append(dataset_json)
        elif dataset_name == "daily_dialog":
            pass
        elif dataset_name == 'olid':
            pass
        else:
            raise NotImplementedError(f"The dataset {dataset_name} is not supported!")

    def __len__(self):
        return len(self.data)
        # return 1

    def __getitem__(self, idx):
        return self.data[idx]


def adversarial_operation(conversation, parse_text, i, **kwargs):
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
        # dataset_json[f] = texts[f][_id]
        if dataset_name == 'boolq':
            filtered_text += " "
    dataset_json['x'] = filtered_text
    dataset_json["y"] = label
    print(dataset_json)

    dataset = CustomInputDataset(dataset_json, dataset_name)

    # def dataset_mapping(x):
    #     return {
    #         "x": x["question"] + x["passage"],
    #         "y": x["label"],
    #     }

    # dataset = dataset.map(function=dataset_mapping)

    model = AutoModelForSequenceClassification.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")
    tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")
    victim = classifiers.TransformersClassifier(model, tokenizer, model.base_model.embeddings.word_embeddings)

    attacker = attackers.PWWSAttacker()
    # prepare for attacking
    attack_eval = AttackEval(attacker, victim)
    # launch attacks and print attack results
    d = attack_eval.eval(dataset, visualize=True)
    # json_list = [d]
    # jsonString = json.dumps(json_list)
    # jsonFile = open("./results.json", "w")
    # jsonFile.write(jsonString)
    # jsonFile.close()
    x_orig = d["x_orig"]
    x_adv = d["x_adv"]
    token_orig = tokenizer.tokenize(x_orig)
    token_adv = tokenizer.tokenize(x_adv)

    # TODO

    widths = [
        (126, 1), (159, 0), (687, 1), (710, 0), (711, 1),
        (727, 0), (733, 1), (879, 0), (1154, 1), (1161, 0),
        (4347, 1), (4447, 2), (7467, 1), (7521, 0), (8369, 1),
        (8426, 0), (9000, 1), (9002, 2), (11021, 1), (12350, 2),
        (12351, 1), (12438, 2), (12442, 0), (19893, 2), (19967, 1),
        (55203, 2), (63743, 1), (64106, 2), (65039, 1), (65059, 0),
        (65131, 2), (65279, 1), (65376, 2), (65500, 1), (65510, 2),
        (120831, 1), (262141, 2), (1114109, 1),
    ]

    def char_width(o):
        if o == 0xe or o == 0xf:
            return 0
        for num, wid in widths:
            if o <= num:
                return wid
        return 1

    def sent_len(s):
        assert isinstance(s, str)
        ret = 0
        for it in s:
            ret += char_width(ord(it))
        return ret

    assert sent_len(token_orig) == sent_len(token_adv)
    curr1 = ""
    curr2 = ""

    for tokenA, tokenB in zip(token_orig, token_adv):
        if tokenA.lower() == tokenB.lower():
            curr1 += tokenA + " "
            curr2 += tokenB + " "
        else:
            curr1 += "<b>" + tokenA + "</b>" + " "
            curr2 += "<b>" + tokenB + "</b>" + " "

    return_s = ''
    return_s += curr1
    return_s += "<br>"
    return_s += curr2

    return return_s, 1
