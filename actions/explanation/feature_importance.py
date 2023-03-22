import json
import torch
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
from transformers import AutoTokenizer

from explained_models.DataLoaderABC.hf_dataloader import HFDataloader
from explained_models.ModelABC.distilbert_qa_boolq import DistilbertQABoolModel
from explained_models.Tokenizer.tokenizer import HFTokenizer
from explained_models.Explainer.explainer import Explainer
import numpy as np


def handle_input(parse_text):
    """
    Handle the parse text and return the list of numbers(ids) and topk value if given
    Args:
        parse_text: parse_text from bot

    Returns: num_list, topk

    """
    num_list = []
    topk = None
    for item in parse_text:
        try:
            if int(item):
                num_list.append(int(item))
        except:
            pass
    if "topk" in parse_text:
        topk = num_list[-1]
        return num_list[:-1], topk
    else:
        return num_list, topk


def get_res(json_list, num_list, topk, tokenizer, num=0):
    """
    Get topk tokens from a single sentence
    Args:
        json_list: data source
        num_list: list of numbers(ids)
        topk: topk value
        tokenizer: for converting input_ids to sentence
        num: current index

    Returns:
        topk tokens
    """
    input_ids = json_list[num_list[num]]["input_ids"]
    explanation = json_list[num_list[num]]["attributions"]
    res = []

    # Get corresponding tokens by input_ids
    tokens = list(tokenizer.decode(input_ids).split(" "))

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
        return_s = f"The <b>topk {topk}</b> important token is <b>{res}.</b>"
    else:
        return_s = f"The <b>topk {topk}</b> important tokens are <b>{res}.</b>"
    return return_s


def feature_importance_operation(conversation, parse_text, i, **kwargs):
    # filter id 5 or filter id 151 or filter id 315 and nlpattribute topk 10 [E]
    # filter id 213 and nlpattribute all [E]
    # filter id 33 and nlpattribute topk 1 [E]
    num_list, topk = handle_input(parse_text)

    data_path = "./cache/boolq/ig_explainer_boolq_explanation.json"
    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")

    if topk is None:
        return "topk is not given", 1
    elif topk >= len(json_list[0]["input_ids"]):
        return "Entered topk is larger than input max length", 1
    else:
        if len(num_list) == 1:
            res = get_res(json_list, num_list, topk, tokenizer, num=0)
            return get_return_str(topk, res), 1
        else:
            return_s = ""
            for num in num_list:
                res = get_res(json_list, num_list, topk, tokenizer, num=num)
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
                # ids = batch['input_ids'][idx_instance]
                # label = batch['labels'][idx_instance]
                # attrbs = attribution[idx_instance]
                # preds = predictions[idx_instance]
                result = {'batch': idx_batch,
                          'instance': idx_instance,
                          'index_running': idx_instance_running,
                          'input_ids': ids,
                          'label': label,
                          'attributions': attrbs,
                          'predictions': preds}
                print(result)
                if store_data:
                    json_list.append(result)
        if store_data:
            jsonString = json.dumps(json_list)
            jsonFile = open(data_path, "w")
            jsonFile.write(jsonString)
            jsonFile.close()


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     model_id = "andi611/distilbert-base-uncased-qa-boolq"
#     tokenizer = HFTokenizer(model_id)
#     dataloader = HFDataloader(tokenizer=tokenizer.tokenizer, batch_size=1, number_of_instance=10)
#     model = DistilbertQABoolModel(dataloader, num_labels=2, model_id=model_id)
#     FeatureAttributionExplainer(model, device=device).generate_explanation()
