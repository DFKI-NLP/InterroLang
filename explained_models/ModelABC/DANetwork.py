import torch
from torch import nn
from transformers import BertForSequenceClassification

DEFAULT_MODEL_ID = "bert-base-uncased"


class DANetwork(nn.Module):
    def __init__(self, bert_emb_size=768, hidden_dim=128, model_id=DEFAULT_MODEL_ID, num_labels=5):
        super(DANetwork, self).__init__()
        self.bert_emb_size = bert_emb_size
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.bert = None
        self.model_id = model_id
        self.create_model()
        self.load_state_dict(torch.load('../da_classifier/saved_model/5e_5e-06lr'))

    def forward(self, input_ids, input_mask):
        output = self.bert(input_ids, attention_mask=input_mask).logits
        return output

    def create_model(self):
        self.bert = BertForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_labels)
