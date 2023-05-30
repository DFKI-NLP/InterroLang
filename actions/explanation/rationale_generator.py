from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM
import pandas as pd
import numpy as np
import torch
import json
import csv
from rationalize import get_few_shot_str
import argparse
import os.path
import sys
from tqdm import tqdm

# directory reach
directory = os.path.dirname(os.path.abspath("__file__"))
print(directory)
# setting path
sys.path.append(directory)
from explained_models.ModelABC.DANetwork import DANetwork
from explained_models.Tokenizer.tokenizer import HFTokenizer
from transformers import GPTNeoXTokenizerFast, GPTNeoXForCausalLM, GPTNeoXConfig

def generate_rationale(dataset_name,write_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt_tokenizer = GPTNeoXTokenizerFast.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
    gpt_model = GPTNeoXForCausalLM.from_pretrained("databricks/dolly-v2-3b")
    gpt_model.to(device)
    instances = []
    explanations = []
    if dataset_name == "boolq":
        model = AutoModelForSequenceClassification.from_pretrained(
            "andi611/distilbert-base-uncased-qa-boolq",
            num_labels=2
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")
        model.to(device)

        dataset = pd.read_csv("data/boolq_validation.csv")

        with open(write_path, 'w',newline='') as file:
            writer = csv.writer(file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Id", "Question", "Passage", "Explanation"])
            for i in range(len(dataset)):
                instances.append([dataset["question"][i], dataset["passage"][i]])
            for idx, instance in tqdm(enumerate(instances), total=len(instances)):
                # if dataset_name == "boolq":
                text = 'Question: ' + instance[0] + '\nPassage: ' + instance[1]
                label_dict = {0: 'false', 1: 'true'}
                text_description = 'question and passage'
                fields = ['question', 'passage']
                fields_enum = ', '.join([f"'{f}'" for f in fields])
                output_description = 'Answer'

                string = instance[0] + ' ' + instance[1]
                instruction = "Please explain the answer: "
                encoding = tokenizer.encode_plus(string, return_tensors='pt', max_length=512, truncation=True)
                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                input_model = {
                    'input_ids': input_ids.long(),
                    'attention_mask': attention_mask.long(),
                }
                output_model = model(**input_model)[0]

                # Get logit
                model_predictions = np.argmax(output_model.cpu().detach().numpy())

                # model_predictions = model.predict()
                pred_str = label_dict[model_predictions]
                # else:
                #     return f"Dataset {dataset_name} currently not supported by rationalize operation", 1
                few_shot_str = get_few_shot_str("cache/boolq/GPT-3.5_rationales_BoolQ_val_400.csv")
                prompt = f"{few_shot_str}"\
                         f"{text}\n" \
                         f"Based on {text_description}, the {output_description} is {pred_str}. " \
                         f"Without using {fields_enum}, or revealing the answer or outcome in your response, " \
                         f"explain why: "

                input_ids = gpt_tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(device)
                generation = gpt_model.generate(
                    input_ids,
                    max_length=2048,
                    no_repeat_ngram_size=2,
                )
                decoded_generation = gpt_tokenizer.decode(generation[0], skip_special_tokens=True)
                #
                #inputs = decoded_generation.split("Based on ")[0]
                explanation = decoded_generation.split("explain why: ")[1]
                writer.writerow([idx,instance[0], instance[1],explanation])
    elif dataset_name == "daily_dialog":
        model = DANetwork()
        tokenizer = HFTokenizer('bert-base-uncased', mode='bert').tokenizer
        #model = AutoModelForSequenceClassification.from_pretrained("./explained_models/da_classifier/saved_model/5e_5e-06lr")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #tokenizer = AutoTokenizer.from_pretrained("./explained_models/da_classifier/saved_model/5e_5e-06lr")
        model.to(device)
        dataset = pd.read_csv("./data/da_test_set_with_indices.csv")
        with open(write_path, 'w',newline='') as file:
            writer = csv.writer(file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Id", "Input Text", "Explanation"])
            for i in range(len(dataset)):
                instances.append(dataset["dialog"][i])
            for idx, instance in tqdm(enumerate(instances), total=len(instances)):
                text = instance
                encoding = tokenizer.encode_plus(text, return_tensors='pt')
                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                input_model = {
                    'input_ids': input_ids.long(),
                    'attention_mask': attention_mask.long(),
                }
                output_model = model(**input_model)[0]

                # Get logit
                model_predictions = np.argmax(output_model.cpu().detach().numpy())

                label_dict = {0: 'dummy', 1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}
                pred = model_predictions
                pred_str = label_dict[pred]
                other_class_names = ", ".join(
                    [label_dict[c] for c in label_dict if label_dict[c] not in [pred, "dummy"]])
                intro = f"The dialogue act of this text has been classified as {pred_str} (over {other_class_names})."
                instruction = "Please explain why: "
                few_shot_str = get_few_shot_str("cache/daily_dialog/GPT-4_rationales_DD_test_200.csv")

                prompt = f"{few_shot_str}" \
                         f"{text}\n" \
                         f"{intro}\n" \
                         f"{instruction}\n"

                input_ids = gpt_tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(device)
                generation = gpt_model.generate(
                    input_ids,
                    max_length=2048,
                    no_repeat_ngram_size=2,
                )
                decoded_generation = gpt_tokenizer.decode(generation[0], skip_special_tokens=True)
                #
                writer.writerow([idx,instance,decoded_generation])
    else:
        model = AutoModelForSequenceClassification.from_pretrained("sinhala-nlp/mbert-olid-en")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("sinhala-nlp/mbert-olid-en")
        model.to(device)
        dataset = pd.read_csv("./data/offensive_val.csv")
        instances = []
        with open(write_path, 'w',newline='') as file:
            writer = csv.writer(file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Id", "InputText", "Explanation"])
            for i in range(len(dataset)):
                instances.append(dataset["text"][i])
            for idx, instance in tqdm(enumerate(instances), total=len(instances)):
                text = "Tweet: '" + instance + "'"
                label_dict = {0: "non-offensive", 1: "offensive"}

                encoding = tokenizer.encode_plus(instance, return_tensors='pt', max_length=512, truncation=True)
                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                input_model = {
                    'input_ids': input_ids.long(),
                    'attention_mask': attention_mask.long(),
                }
                output_model = model(**input_model)[0]

                # Get logit
                model_predictions = np.argmax(output_model.cpu().detach().numpy())

                pred_str = label_dict[model_predictions]
                intro = f"The tweet has been classified as {pred_str}."
                instruction = "Please explain why: "
                few_shot_str = get_few_shot_str("cache/olid/GPT-4_rationales_OLID_val_132.csv")

                prompt = f"{few_shot_str}" \
                         f"{text}\n" \
                         f"{intro}\n" \
                         f"{instruction}\n"

                input_ids = gpt_tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(device)
                generation = gpt_model.generate(
                    input_ids,
                    max_length=2048,
                    no_repeat_ngram_size=2,
                )
                decoded_generation = gpt_tokenizer.decode(generation[0], skip_special_tokens=True)
                #
                # inputs = decoded_generation.split("Based on ")[0]
                #explanation = decoded_generation.split("explain why: ")[1]
                writer.writerow([idx, instance, decoded_generation])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--write_path', type=str, required=True)
    args = parser.parse_args()
    generate_rationale(args.dataset_name, args.write_path)
