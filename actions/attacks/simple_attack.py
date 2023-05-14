import OpenAttack
from OpenAttack.tags import Tag, TAG_English
from OpenAttack.metric.selectors.base import MetricSelector

from actions.attacks.attack_eval import *
from actions.attacks.bert_attack import *
from actions.attacks.vis_util import *
from actions.attacks.punct_tokenizer import *

import torch
import string
import datasets

import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class CustomClassifier(OpenAttack.Classifier):
    def __init__(self, model, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def get_prob(self, sentences):
        with torch.no_grad():
            tokenized = self.tokenizer(sentences, pad_to_max_length=True, padding="max_length", max_length=256, truncation=True, add_special_tokens=True, return_tensors="pt")
            probs = self.softmax(self.model(**tokenized).logits.cpu()).numpy()
            return probs
    
    def get_pred(self, sentences):
        return self.get_prob(sentences).argmax(axis=1)


def attack_operation(conversation, parse_text, i, **kwargs):
    model = conversation.get_var("model").contents
    parsed_id = ""
    return_s = ""
    if len(parse_text)>0:
        parsed_id = parse_text[i+1]
    while parsed_id[-1] in string.punctuation:
        parsed_id = parsed_id[:-1]
    if parsed_id.isdigit():
        id_val = int(parsed_id)
    else:
        return "Sorry, invalid id", 1    

    dataset = conversation.get_var("dataset").contents["dataset_name"]
    x_data = conversation.get_var("dataset").contents["X"]
    if dataset=="boolq":
        question = x_data.iloc[id_val]["question"]
        passage = x_data.iloc[id_val]["passage"]
        instance_to_predict = {"question":question, "passage":passage}
    else:
        instance = x_data.iloc[id_val]["text"]
        instance_to_predict = instance
    
    gold_label = conversation.get_var("dataset").contents["y"].iloc[id_val]
    transformer_model = conversation.get_var("model").contents
    model = transformer_model.model
    tokenizer = transformer_model.tokenizer
    victim = CustomClassifier(model, tokenizer)
    attacker = BERTAttacker(mlm_path="bert-base-uncased")
    dataset = datasets.Dataset.from_dict({
        "x": [
            question + " [sep] " + passage
        ],
        "y": [
            gold_label
        ]
    })
    oa_punct_tokenizer = PunctTokenizer()
    attack_eval = AttackEval(attacker, victim, tokenizer = oa_punct_tokenizer)
    res = attack_eval.eval(dataset, visualize=True)    
    x_orig = res["x_orig"]
    x_adv = res["x_adv"]
    change1, change2 = get_change(x_orig, x_adv)    
    return_s += "<b>Original text:</b><br>" + change1.replace("[sep]","<br>Passage: ")
    return_s += "<br><br><b>Adversarial text:</b><br>" + change2.replace("[sep]","<br>Passage: ")

    max_orig_prob = np.max(res["y_orig_probs"])
    max_orig_idx = np.argmax(res["y_orig_probs"])
    max_adv_prob = np.max(res["y_adv_probs"])
    max_adv_idx = np.argmax(res["y_adv_probs"])
    id2label = conversation.get_var("dataset").contents["id2label"]

    return_s += "<br><br>Prediction flipped from <b><font color='green'>"+str(id2label[max_orig_idx])+"</font></b> ("+str(round(max_orig_prob,2))+") to <b><font color='red'>"+str(id2label[max_adv_idx])+"</font></b> ("+str(round(max_adv_prob,2))+")"
    return return_s, 1

