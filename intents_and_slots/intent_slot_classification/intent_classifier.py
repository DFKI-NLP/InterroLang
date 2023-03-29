import datasets
from datasets import load_dataset
from transformers import AutoAdapterModel, AutoTokenizer, AutoConfig, BertConfig, BertModelWithHeads, AdapterConfig
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import nn
import copy
import sys
import os

from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"

task = "similar" # "predict", "include", "nlpcfe", "similar"

do_training = True
batch_size = 32
model_name = "bert-base-cased"
data_folder = "csv_intents/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAdapterModel.from_pretrained(model_name)

def encode_data(data):
    encoded = tokenizer([doc for doc in data["text"]], pad_to_max_length=True, padding="max_length", max_length=128, truncation=True, add_special_tokens=True)
    return (encoded)


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

# training the model

if do_training:
    config = AdapterConfig.load("pfeiffer")
    model.add_adapter(task, config=config)

    model.add_classification_head(task, num_labels=2, use_pooler=True)
    model.train_adapter(task)

    train_task_dataset = datasets.Dataset.from_csv(data_folder+"/"+task+"_train.csv")
    train_task_dataset = train_task_dataset.map(encode_data, batched=True, batch_size=batch_size)
    #train_task_dataset = train_task_dataset.rename_column("tokens","text").rename_column("tags","labels")

    dev_task_dataset = datasets.Dataset.from_csv(data_folder+"/"+task+"_dev.csv")
    dev_task_dataset = dev_task_dataset.map(encode_data, batched=True, batch_size=batch_size)

    train_task_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_task_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model.to(device)


    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=8,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=100,
        output_dir="training_output",
        overwrite_output_dir=True,
        remove_unused_columns=False,
    )


    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_task_dataset,
        eval_dataset=dev_task_dataset,
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    print(trainer.evaluate())

    model.save_adapter("adapters/"+task, task)
    #model.save_head("heads_saved/"+task+"_head/", task+"_head")


# test evaluation

from transformers import TextClassificationPipeline
intexts = []
gold_labels = []
test_task_dataset = datasets.Dataset.from_csv(data_folder+"/"+task+"_test.csv")
for i in range(len(test_task_dataset)):
    intexts.append(test_task_dataset["text"][i])
    gold_labels.append(test_task_dataset["labels"][i])


adapter = model.load_adapter("adapters/"+task)
head = model.load_head("adapters/"+task)

model.active_adapters = adapter
dact_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, task=task, device=0)

tp = 0
fp = 0
fn = 0
for i, intext in enumerate(intexts):
    predicted_label = ""
    prediction = dact_classifier(intext)
    if len(prediction)>0:
        predicted_label = int(prediction[0]["label"].replace("LABEL_",""))
    gold_label = int(gold_labels[i])
    
    if predicted_label==gold_label:
        tp+=1
    elif predicted_label==0:
        fn+=1
    else:
        fp+=1

f1score = 0
if tp+fp>0:
    prec = tp/(tp+fp)
else:
    prec = 0
if tp+fn>0:
    rec = tp/(tp+fn)
else:
    rec = 0
if prec+rec>0:
    f1score = 2*prec*rec/(prec+rec)
else:
    f1score = 0
print("F1:", round(f1score,3))

