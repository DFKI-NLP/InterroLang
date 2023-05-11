import json

import torch
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from explained_models.Explainer.explainer import Explainer
import numpy as np

from utils.custom_input import generate_explanation

from nltk.tokenize import sent_tokenize


def handle_input(parse_text):
    """
    Handle the parse text and return the list of numbers(ids) and topk value if given
    Args:
        parse_text: parse_text from bot

    Returns: id_list, topk

    """
    id_list = []
    topk = None

    for item in parse_text:
        try:
            if int(item):
                if int(item) > 0:
                    id_list.append(int(item))
        except:
            pass

    if "topk" in parse_text:
        if len(id_list) >= 1:
            topk = id_list[-1]

        # filter id 5 or filter id 151 or filter id 315 and nlpattribute topk 10 [E]
        if len(id_list) > 1:
            return id_list[:-1], topk
        else:
            # nlpattribute topk 3 [E]
            return None, topk
    else:
        if len(id_list) >= 1:
            # filter id 213 and nlpattribute all [E]
            if "all" in parse_text:
                return id_list, 1

            # filter id 213 and nlpattribute sentence [E]
            if "sentence" in parse_text:
                return id_list, -1

        # nlpattribute [E]
        return id_list, topk


def get_res(json_list, topk, tokenizer, num=0):
    """
    Get topk tokens from a single sentence
    Args:
        json_list: data source
        topk: topk value
        tokenizer: for converting input_ids to sentence
        num: current index

    Returns:
        topk tokens
    """
    input_ids = json_list[num]["input_ids"]
    explanation = json_list[num]["attributions"]
    res = []

    # Get corresponding tokens by input_ids
    # tokens = list(tokenizer.decode(input_ids).split(" "))
    tokens = list(tokenizer.convert_ids_to_tokens(input_ids))

    idx = np.argsort(explanation)[::-1][:topk]

    for i in idx:
        res.append(tokens[i])

    return_s = ', '.join(i for i in res)
    return return_s


def get_return_str(topk, res):
    """
    Generate return string using template
    Args:
        topk: topk value
        res:  topk tokens

    Returns:
        object: template string
    """
    if topk == 1:
        return_s = f"The <b>most</b> important token is <b>{res}.</b>"
    else:
        return_s = f"The <b>{topk} most</b> important tokens are <b>{res}.</b>"
    return return_s



def get_explanation(dataset_name, inputs, file_name="sentence_level"):
    if dataset_name == "boolq":
        model = AutoModelForSequenceClassification.from_pretrained("andi611/distilbert-base-uncased-qa-boolq",
                                                                   num_labels=2)
    elif dataset_name == "daily_dialog":
        pass
    elif dataset_name == "olid":
        model = AutoModelForSequenceClassification.from_pretrained("sinhala-nlp/mbert-olid-en")
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not supported!")

    res_list = generate_explanation(model, dataset_name, inputs, file_name=file_name)

    return res_list


def explanation_with_custom_input(conversation, topk):
    """
    Get explanation of custom inputs from users
    Args:
        conversation: conversation object
        topk: most top k important tokens

    Returns:
        formatted string
    """

    inputs = [conversation.custom_input]

    if len(inputs) == 0:
        return None

    dataset_name = conversation.describe.get_dataset_name()

    res_list = get_explanation(dataset_name, inputs)
    return_s = ""
    for res in res_list:
        if dataset_name == "boolq":
            original_text = res["text"]

            return_s += "The original text is:  <br>"
            return_s += "<i>"
            return_s += original_text
            return_s += "</i>"
            return_s += "<br><br>"

            text = "[CLS] " + original_text + " [SEP]"
            attr = res["attributions"]
            text_list = text.split()

            # Get indices according to absolute attribution scores ascending
            idx = np.argsort(np.absolute(np.copy(attr)))

            # Get topk tokens
            topk_tokens = []
            for i in np.argsort(attr)[-topk:][::-1]:
                topk_tokens.append(text_list[i])

            score_ranking = []
            for i in range(len(idx)):
                score_ranking.append(list(idx).index(i))
            fraction = 1.0 / (len(text_list) - 1)

            return_s += f"Top {topk} token(s): "
            for i in topk_tokens:
                return_s += f"<b>{i}</b>"
                return_s += " "
            return_s += '<br>'

            return_s += "The visualization: "
            for i in range(1, len(text_list) - 1):
                if attr[i] >= 0:
                    # Assign red to tokens with positive attribution
                    return_s += f"<span style='background-color:rgba(255,0,0,{round(fraction * score_ranking[i], conversation.rounding_precision)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
                else:
                    # Assign blue to tokens with negative attribution
                    return_s += f"<span style='background-color:rgba(0,0,255,{round(fraction * score_ranking[i], conversation.rounding_precision)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
                return_s += text_list[i]
                return_s += "</span>"
                return_s += ' '

            return_s += '<br><br><br>'
        elif dataset_name == "olid":
            original_text = res["text"]
            _input = res["original_text"]

            return_s += "The original text is:  <br>"
            return_s += "<i>"
            return_s += _input
            return_s += "</i>"
            return_s += "<br><br>"

            attr = res["attributions"]

            assert len(attr) == len(original_text)

            # Get indices according to absolute attribution scores ascending
            idx = np.argsort(np.absolute(np.copy(attr)))

            # Get topk tokens
            topk_tokens = []
            # print(text_list)
            for i in np.argsort(attr)[-topk:][::-1]:
                print("i: ", i)
                topk_tokens.append(original_text[i])

            score_ranking = []
            for i in range(len(idx)):
                score_ranking.append(list(idx).index(i))
            fraction = 1.0 / (len(original_text) - 1)

            return_s += f"Top {topk} token(s): "
            for i in topk_tokens:
                return_s += f"<b>{i}</b>"
                return_s += " "
            return_s += '<br>'

            return_s += "The visualization: "
            # for i in range(1, len(text_list) - 1):
            for i in range(1, len(original_text) - 1):
                if attr[i] >= 0:
                    # Assign red to tokens with positive attribution
                    return_s += f"<span style='background-color:rgba(255,0,0,{round(fraction * score_ranking[i], conversation.rounding_precision)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
                else:
                    # Assign blue to tokens with negative attribution
                    return_s += f"<span style='background-color:rgba(0,0,255,{round(fraction * score_ranking[i], conversation.rounding_precision)});padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone'>"
                # return_s += text_list[i]
                return_s += original_text[i]
                return_s += "</span>"
                return_s += ' '

            return_s += '<br><br><br>'

    return return_s


def get_sentence_level_feature_importance(conversation, sentences):
    # sentences = parse_text[i+1]
    inputs = sent_tokenize(sentences)
    dataset_name = conversation.describe.get_dataset_name()
    res_list = get_explanation(dataset_name, inputs, file_name="sentence_level")

    return_s = f'The original text is: <i>{sentences}</i> <br><br>'
    counter = 1

    for res in res_list:
        attr = res["attributions"]

        if dataset_name == 'boolq':
            text = res["text"]
        elif dataset_name == 'olid':
            text = res["original_text"]

        return_s += "<ul>"
        return_s += "<li>"
        return_s += f"Sentence {counter}: <i>{text}</i>"
        return_s += "</li>"

        return_s += "<li>"
        return_s += f"Average saliency score: <b>{round(sum(attr) / len(attr), conversation.rounding_precision)}</b>"
        return_s += "</li>"

        return_s += "<li>"
        return_s += f"Prediction: <b>{conversation.class_names[res['predictions']]}</b>"
        return_s += "</li>"
        return_s += "</ul>"
        counter += 1
    return return_s


def feature_importance_operation(conversation, parse_text, i, **kwargs):
    # filter id 5 or filter id 151 or filter id 315 and nlpattribute topk 10 [E]
    # filter id 213 and nlpattribute all [E]
    # filter id 33 and nlpattribute topk 1 [E]

    id_list, topk = handle_input(parse_text)

    if topk is None:
        topk = 3

    # If id is not given

    if conversation.used is False and conversation.custom_input is not None:
        if "sentence" in parse_text:
            return_s = get_sentence_level_feature_importance(conversation, sentences=conversation.custom_input)
            return return_s, 1
        else:
            explanation = explanation_with_custom_input(conversation, topk)
            return explanation, 1

    if topk == -1:
        return_s = ''
        for _id in id_list:
            f_names = list(conversation.temp_dataset.contents['X'].columns)
            texts = conversation.temp_dataset.contents['X']
            filtered_text = ''

            # Get the first column, also for boolq, we only need question column not passage
            for f in f_names[:1]:
                filtered_text += texts[f][_id]
                filtered_text += " "
            return_s += f'ID {_id}: '
            return_s += get_sentence_level_feature_importance(conversation, sentences=filtered_text)
            return_s += '<br><br>'
        return return_s, 1

    # Get the dataset name
    name = conversation.describe.get_dataset_name()

    data_path = f"./cache/{name}/ig_explainer_{name}_explanation.json"
    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    if name == 'boolq':
        tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")
    elif name == "daily_dialog":
        pass
    else:
        tokenizer = AutoTokenizer.from_pretrained("sinhala-nlp/mbert-olid-en")

    if topk >= len(json_list[0]["input_ids"]):
        return "Entered topk is larger than input max length", 1
    else:
        if len(id_list) == 1:
            res = get_res(json_list, topk, tokenizer, num=id_list[0])
            return get_return_str(topk, res), 1
        else:
            return_s = ""
            for num in id_list:
                res = get_res(json_list, topk, tokenizer, num=num)
                temp = get_return_str(topk, res)
                return_s += f"For id {num}: {temp}"
                return_s += "<br>"
            return return_s, 1


class FeatureAttributionExplainer(Explainer):
    def __init__(self, model, device):
        super(Explainer).__init__()
        self.device = device
        self.model = model.model.to(device)
        # self.tokenizer = model.tokenizer
        self.dataloader = model.dataloader
        self.forward_func = self.get_forward_func()
        self.explainer = LayerIntegratedGradients(forward_func=self.forward_func,
                                                  layer=self.get_embedding_layer())
        self.pad_token_id = self.model.tokenizer.pad_token_id

    def get_forward_func(self):
        # TODO: Implement forward functions for non-BERT models (LSTM, ...)
        def bert_forward(input_ids, attention_mask):
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
            }
            output_model = self.model(**input_model)[0]
            return output_model

        return bert_forward

    def get_embedding_layer(self):
        return self.model.base_model.embeddings

    @staticmethod
    def get_inputs_and_additional_args(base_model, batch):
        assert 'input_ids' in batch, f'Input ids expected for {base_model} but not found.'
        assert 'attention_mask' in batch, f'Attention mask expected for {base_model} but not found.'
        input_ids = batch['input_ids']
        additional_forward_args = (batch['attention_mask'])
        return input_ids, additional_forward_args

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def get_baseline(self, batch):
        assert 'special_tokens_mask' in batch
        if self.pad_token_id == 0:
            # all non-special token ids are replaced by 0, the pad id
            baseline = batch['input_ids'] * batch['special_tokens_mask']
            return baseline
        else:
            baseline = batch['input_ids'] * batch['special_tokens_mask']  # all input ids now 0
            # add pad_id everywhere,
            # substract again where special tokens are, leaves non special tokens with pad id
            # and conserves original pad ids
            baseline = (baseline + self.pad_token_id) - (batch['special_tokens_mask'] * self.pad_token_id)
            return baseline

    def compute_feature_attribution_scores(
            self,
            batch
    ):
        r"""
        :param batch
        :return:
        """
        self.model.eval()
        self.model.zero_grad()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(
            base_model=type(self.model.base_model),
            batch=batch
        )
        predictions = self.forward_func(
            inputs,
            *additional_forward_args
        )
        pred_id = torch.argmax(predictions, dim=1)
        baseline = self.get_baseline(batch=batch)
        attributions = self.explainer.attribute(
            inputs=inputs,
            n_steps=50,
            additional_forward_args=additional_forward_args,
            target=pred_id,
            baselines=baseline,
            internal_batch_size=1,
        )
        attributions = torch.sum(attributions, dim=2)
        return attributions, predictions

    def generate_explanation(self, store_data=False, data_path="../../cache/boolq/ig_explainer_boolq_explanation.json"):
        def detach_to_list(t):
            return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t

        if store_data:
            json_list = []

        for idx_batch, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), position=0, leave=True):
            if idx_batch % 1000 == 0:
                print(f'(Progress) Processing batch {idx_batch} / instance {idx_batch * self.dataloader.batch_size}')
            attribution, predictions = self.compute_feature_attribution_scores(batch)

            for idx_instance in range(len(batch['input_ids'])):
                idx_instance_running = (idx_batch * self.dataloader.batch_size)

                ids = detach_to_list(batch['input_ids'][idx_instance])
                label = detach_to_list(batch['labels'][idx_instance])
                attrbs = detach_to_list(attribution[idx_instance])
                preds = detach_to_list(predictions[idx_instance])
                result = {'batch': idx_batch,
                          'instance': idx_instance,
                          'index_running': idx_instance_running,
                          'input_ids': ids,
                          'label': label,
                          'attributions': attrbs,
                          'predictions': preds}
                if store_data:
                    json_list.append(result)
        if store_data:
            jsonString = json.dumps(json_list)
            jsonFile = open(data_path, "w")
            jsonFile.write(jsonString)
            jsonFile.close()
