import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm

MAPPING = {0: "False", 1: "True"}


class Dataset:
    def __init__(self, texts, labels, tokenizer):

        self.tokenizer = tokenizer

        self.max_seq_length = 256
        self.input_ids = []
        self.input_masks = []
        self.labels = []

        for i in range(len(texts)):
            sample_text, label = texts[i], labels[i]
            ids, mask = self.get_id_with_mask(sample_text)
            self.input_ids.append(ids)
            self.input_masks.append(mask)
            self.labels.append(torch.tensor(label))

    def get_id_with_mask(self, input_text):
        encoded_dict = self.tokenizer.encode_plus(
            input_text.lower(),
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def __getitem__(self, idx):
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        return self.input_ids[idx].squeeze(), self.input_masks[idx].squeeze(), label

    def __len__(self):
        return len(self.input_ids)


def get_embedding_layer(model):
    return model.bert.embeddings


def get_inputs_and_additional_args(batch):
    input_ids = batch[0]
    additional_forward_args = (batch[1].to(device))
    return input_ids, additional_forward_args


def get_forward_func():
    def bert_forward(input_ids, attention_masks):
        # adapt to input mask and enlarge the dimension of input_mask
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        input_model = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_masks.long()[None, :],
        }
        output_model = model(**input_model)
        return output_model.logits

    return bert_forward


def compute_feature_attribution_scores(batch, model, device='cpu'):
    model.to(device)
    model.eval()
    model.zero_grad()
    inputs, additional_forward_args = get_inputs_and_additional_args(
        batch=batch
    )
    inputs.to(device)

    forward_func = get_forward_func()
    predictions = forward_func(
        inputs,
        *additional_forward_args
    )
    predictions.to(device)
    pred_id = torch.argmax(predictions, dim=1)
    pred_id.to(device)

    baseline = torch.zeros(batch[0].shape)
    # speicial_tokens_mask = batch[0] * 0
    # speicial_tokens_mask[0][0] = 1
    # speicial_tokens_mask[0][-1] = 1
    # baseline = batch[0] * speicial_tokens_mask
    baseline.to(device)

    explainer = LayerIntegratedGradients(forward_func=forward_func,
                                         layer=get_embedding_layer(model))

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


if __name__ == '__main__':
    data_path = "/netscratch/qwang/offenseval_val.csv"
    res_path = "/netscratch/qwang/ig_explainer_olid_explanation.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(data_path)
    texts = df["text"]
    labels = df["label"]

    tokenizer = AutoTokenizer.from_pretrained("sinhala-nlp/mbert-olid-en")

    model = AutoModelForSequenceClassification.from_pretrained("sinhala-nlp/mbert-olid-en")
    model.to(device=device)

    dataset = Dataset(texts, labels, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset)
    json_list = []
    for idx_batch, b in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True):
        b[0].to(device)
        b[1].to(device)

        attribution, predictions = compute_feature_attribution_scores(b, model, device)
        idx_instance_running = idx_batch

        ids = detach_to_list(b[0][0])
        attrbs = detach_to_list(attribution[0])
        preds = detach_to_list(predictions[0])
        result = {'batch': idx_batch,
                  # 'instance': idx_instance,
                  'index_running': idx_instance_running,
                  'input_ids': ids,
                  'label': b[2][0].item(),
                  'attributions': attrbs,
                  'predictions': preds
                  }
        json_list.append(result)

    jsonString = json.dumps(json_list)
    jsonFile = open(res_path, "w")
    jsonFile.write(jsonString)
    jsonFile.close()
