import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
from explained_models.da_classifier.da_model_utils import DANetwork, DADataset

MAPPING = {0: "__dummy__" , 1: "inform", 2: "question", 3: "directive", 4: "commissive"}


def get_embedding_layer(model):
    return model.bert.base_model.embeddings


def get_inputs_and_additional_args(batch):
    # assert 'input_ids' in batch, f'Input ids expected for {base_model} but not found.'
    # assert 'attention_mask' in batch, f'Attention mask expected for {base_model} but not found.'
    input_ids = batch[0]
    additional_forward_args = (batch[1].to(device))
    return input_ids, additional_forward_args


def get_forward_func():
    def bert_forward(input_ids, attention_masks):
        # adapt to input mask and enlarge the dimension of input_mask
        input_ids.to(device)
        attention_masks.to(device)
        input_model = {
            'input_ids': input_ids.long(),
            'input_mask': attention_masks.long()[None, :],
        }
        output_model = da(**input_model)
        output_model.to(device)
        return output_model

    return bert_forward


def compute_feature_attribution_scores(batch, model, device):
    model.to(device)
    model.eval()
    model.zero_grad()
    # batch = {k: v.to(device) for k, v in batch.items()}
    inputs, additional_forward_args = get_inputs_and_additional_args(
        # base_model=type(self.model.base_model),
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store_data = True
    data_path = "/netscratch/qwang/ig_explainer_daily_dialog_explanation.json"

    if store_data:
        json_list = []

    da = DANetwork()
    da.load_state_dict(torch.load('/netscratch/qwang/5e_5e-06lr'))
    da.to(device=device)
    test_dataloader = torch.load('/netscratch/qwang/test_dataloader.pth')

    for idx_batch, b in tqdm(enumerate(test_dataloader), total=len(test_dataloader), position=0, leave=True):
        # if idx_batch % 1000 == 0:
        #     print(f'(Progress) Processing batch {idx_batch} / instance {idx_batch * 1}')
        b[0].to(device)
        b[1].to(device)

        attribution, predictions = compute_feature_attribution_scores(b, da, device)
        idx_instance_running = idx_batch

        ids = detach_to_list(b[0][0])
        # label = b['labels'][idx_instance]
        attrbs = detach_to_list(attribution[0])
        # preds = predictions[idx_instance]
        # ids = batch['input_ids'][idx_instance]
        # label = batch['labels'][idx_instance]
        # attrbs = attribution[idx_instance]
        preds = torch.argmax(predictions, dim=1)
        result = {'batch': idx_batch,
                  # 'instance': idx_instance,
                  'index_running': idx_instance_running,
                  'input_ids': ids,
                  'label': MAPPING[b[2][0].item()],
                  'attributions': attrbs,
                  'predictions': preds
                  }
        print(result)
        if store_data:
            json_list.append(result)
    if store_data:
        jsonString = json.dumps(json_list)
        jsonFile = open(data_path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

