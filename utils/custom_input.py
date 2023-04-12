import json
import os

import torch
from captum.attr import LayerIntegratedGradients
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomInputDataset(Dataset):
    def __init__(self, inputs, dataset_name):
        self.data = []

        if dataset_name == "boolq":
            self.tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")

            for string in inputs:
                encoding = self.tokenizer.encode_plus(string, add_special_tokens=True, return_tensors='pt')
                input_ids = encoding["input_ids"][0]
                attention_mask = encoding["attention_mask"][0]
                input_model = {
                    'input_ids': input_ids.long(),
                    'attention_mask': attention_mask.long(),
                }
                self.data.append(input_model)
        elif dataset_name == "daily_dialog":
            pass
        elif dataset_name == 'olid':
            pass
        else:
            raise NotImplementedError(f"The dataset {dataset_name} is not supported!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(inputs, dataset_name, batch_size=1):
    customInputDataset = CustomInputDataset(inputs, dataset_name)
    customDataLoader = DataLoader(dataset=customInputDataset, batch_size=batch_size)
    return customDataLoader


def get_embedding_layer(model, dataset_name):
    if dataset_name == 'boolq':
        return model.base_model.embeddings
    elif dataset_name == 'daily_dialog':
        return model.bert.base_model.embeddings
    else:
        return model.bert.embeddings


def get_inputs_and_additional_args(batch, dataset_name):
    if dataset_name == 'boolq':
        input_ids = batch['input_ids']
        additional_forward_args = (batch['attention_mask'].to(device))
    elif dataset_name == "daily_dialog":
        input_ids = batch[0]
        additional_forward_args = (batch[1].to(device))
    else:
        input_ids = batch[0]
        additional_forward_args = (batch[1].to(device))

    return input_ids, additional_forward_args


def get_forward_func(dataset_name, model):
    def bert_forward(input_ids, attention_masks):
        # adapt to input mask and enlarge the dimension of input_mask
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        if dataset_name == 'boolq':
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_masks.long()[None, :],
            }
        elif dataset_name == "daily_dialog":
            input_model = {
                'input_ids': input_ids.long(),
                'input_mask': attention_masks.long()[None, :],
            }
        else:
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_masks.long()[None, :],
            }

        output_model = model(**input_model)
        return output_model.logits

    return bert_forward


def compute_feature_attribution_scores(batch, model, dataset_name):
    model.to(device)
    model.eval()
    model.zero_grad()

    inputs, additional_forward_args = get_inputs_and_additional_args(
        batch=batch, dataset_name=dataset_name
    )
    inputs.to(device)

    forward_func = get_forward_func(dataset_name, model)
    predictions = forward_func(
        inputs,
        *additional_forward_args
    )
    pred_id = torch.argmax(predictions, dim=1)

    # baseline = torch.zeros(batch["input_ids"].shape)

    if dataset_name == 'boolq':
        special_tokens_mask = batch["input_ids"] * 0
        special_tokens_mask[0][0] = 1
        special_tokens_mask[0][-1] = 1
        baseline = batch["input_ids"] * special_tokens_mask
    elif dataset_name == 'daily_dialog':
        special_tokens_mask = batch[0] * 0
        special_tokens_mask[0][0] = 1
        special_tokens_mask[0][-1] = 1
        baseline = batch[0] * special_tokens_mask
    else:
        special_tokens_mask = batch[0] * 0
        special_tokens_mask[0][0] = 1
        special_tokens_mask[0][-1] = 1
        baseline = batch[0] * special_tokens_mask

    explainer = LayerIntegratedGradients(forward_func=forward_func,
                                         layer=get_embedding_layer(model, dataset_name))

    attributions = explainer.attribute(
        inputs=inputs,
        n_steps=50,
        additional_forward_args=additional_forward_args,
        target=pred_id,
        baselines=baseline,
        internal_batch_size=1,
    )
    attributions = torch.sum(attributions, dim=2).to(device)
    return attributions, predictions


def detach_to_list(t):
    return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t


def generate_explanation(model, dataset_name, inputs):
    print(device)

    cache_path = f"./cache/{dataset_name}/{dataset_name}_custom_input_explanation.json"

    if os.path.exists(cache_path):
        fileObject = open(cache_path, "r")
        jsonContent = fileObject.read()
        res_list = json.loads(jsonContent)

        if len(inputs) == 1:
            if res_list["text"] == inputs[0]:
                return res_list
        else:
            cache_text = [i["text"] for i in res_list]
            cache_text_set = set(cache_text)

            # If cache contains all inputs
            if set(inputs).issubset(cache_text_set):
                json_list = []
                for i in res_list:
                    if i["text"] in inputs:
                        json_list.append(i)
                return json_list

    dataloader = get_dataloader(inputs, dataset_name)

    json_list = []

    model.to(device=device)

    if dataset_name == 'boolq':
        for idx_batch, b in enumerate(dataloader):
            attribution, predictions = compute_feature_attribution_scores(b, model, dataset_name)

            # ids = detach_to_list(b)
            # label = b['labels'][idx_instance]
            attrbs = detach_to_list(attribution[0])
            # preds = predictions[idx_instance]
            # ids = batch['input_ids'][idx_instance]
            # label = batch['labels'][idx_instance]
            # attrbs = attribution[idx_instance]
            preds = torch.argmax(predictions, dim=1)
            result = {
                'input_ids': detach_to_list(b["input_ids"]),
                "text": inputs[idx_batch],
                'attributions': attrbs,
                'predictions': preds.item()
            }
            json_list.append(result)

    # Store the chat history
    if not os.path.exists(cache_path):
        jsonString = json.dumps(json_list)
        jsonFile = open(cache_path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()
    else:
        fileObject = open(cache_path, "r")
        jsonContent = fileObject.read()
        res_list = json.loads(jsonContent)
        res_list += json_list

        # Append the new result in json
        jsonString = json.dumps(res_list)
        jsonFile = open(cache_path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    return json_list
