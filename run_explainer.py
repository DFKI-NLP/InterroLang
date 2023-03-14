import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from boolq.data import get_dataset
from feature_importance.ig_explainer import FeatureAttributionExplainer


def detach_to_list(t):
    return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t


device = "cuda"
model_id = "andi611/distilbert-base-uncased-qa-boolq"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.tokenizer = tokenizer
explainer = FeatureAttributionExplainer(model=model, device=device)
explainer.to(device)
batch_size = 1
dataset = get_dataset(tokenizer)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

for idx_batch, batch in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True):
    if idx_batch % 1000 == 0:
        print(f'(Progress) Processing batch {idx_batch} / instance {idx_batch * batch_size}')
    attribution, predictions = explainer.compute_feature_attribution_scores(batch)

    for idx_instance in range(len(batch['input_ids'])):
        idx_instance_running = (idx_batch * batch_size)

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
        print(result)
