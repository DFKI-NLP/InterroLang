import json
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional
import random

import torch
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass(frozen=True)
class InputExample:

    prompt: str
    explanation: str


class TrainingDataset(Dataset):
    features: List[InputExample]

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputExample:
        return self.features[i]


def load_raw_dataset(split, args):
    #data_path = os.path.join('./data', args.dataset, '{}.jsonl'.format(split))
    #ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    #print
    CONFIG_PATH = '../cache/boolq/GPT-3.5_rationales_BoolQ_val_400.csv'
    data = pd.read_csv(CONFIG_PATH)
    train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)

    dataset = []

    for example_id, line in tqdm(train.iterrows(), desc='processing {}'.format(split)):
           example = line
           dataset.append(
                    InputExample(
                        prompt=example["prompt"],
                        explanation=example["completion"],

                    )
             )

    for example in dataset[:2]:
        print("*** Example ***")
        print(example)

    return TrainingDataset(dataset)


def get_label_tensor(raw_label, tokenizer, args):
    label_ids = tokenizer.encode(raw_label, add_special_tokens=False)
    label_ids = label_ids[:args.max_dec_length]
    label_ids += [-100] * (args.max_dec_length - len(label_ids))
    return label_ids


def format_input(question, choices=None):
    input_seq = "Question: {}".format(question.strip())
    # input_seq += " Answer: {}.".format(choice.strip())
    if choices is not None:
        input_seq += " Answer Choices:"
        for choice_id, choice in enumerate(choices):
            input_seq += " ({}) {}".format(chr(ord('a') + choice_id), choice)
        input_seq += '.'
    return input_seq


def format_explanation(explanation):
    input_seq = ' Explanation: ' + explanation.strip()
    return input_seq


class Data_Collator_for_Training(object):
    def __init__(self, tokenizer, args, mask_inference=False, dropout_context=0):
        self.tokenizer = tokenizer
        self.mask_inference = mask_inference
        self.dropout_context = dropout_context
        self.args = args

    def __call__(self, examples):

        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        decoder_label_tensor = []
        label_tensor = []
        smoothing_tensor = []

        for example_idx, example in enumerate(examples):
            input_ids = []
            attention_mask = []

            context = example.prompt
            input_ids = self.tokenizer.encode(context.strip(), add_special_tokens=False)
            explanation = example.explanation
            added_ids = self.tokenizer.encode(explanation,
                                              add_special_tokens=False)

            encoder_input_tensor.append(input_ids)

            encoder_attention_mask_tensor.append([1]*len(input_ids[:self.args.max_enc_length]) + [0] * (self.args.max_enc_length - len(input_ids)))
            decoder_label_tensor.append(added_ids)
        choices_input_ids = [ids[:self.args.max_enc_length] for ids in encoder_input_tensor]
        choices_input_ids = [ids + [self.tokenizer.pad_token_id] * (self.args.max_enc_length - len(ids)) for ids in
                             choices_input_ids]
        label_tensor = [ids[:self.args.max_dec_length] for ids in decoder_label_tensor]
        label_tensor = [ids + [self.tokenizer.pad_token_id] * (self.args.max_dec_length - len(ids))
                        for ids in label_tensor]
        return tuple(torch.tensor(t) for t in
                     [choices_input_ids, encoder_attention_mask_tensor, label_tensor])


def get_tensor_dataset(split, tokenizer, args):
    #data_path = os.path.join('./data', args.dataset, '{}.jsonl'.format(split))
    #ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    CONFIG_PATH = '../cache/boolq/GPT-3.5_rationales_BoolQ_val_400.csv'
    data = pd.read_csv(CONFIG_PATH)
    train, test = train_test_split(data, test_size=0.3,random_state=42, shuffle=True)
    dev,test = train_test_split(test, test_size=0.2, random_state=42, shuffle=True)
    split_args = dev if split == 'dev' else test
    encoder_input_tensor = []
    encoder_attention_mask_tensor = []
    decoder_label_tensor = []
    task_label_tensor = []
    for example_idx, example in tqdm(split_args.iterrows(), desc='processing {}'.format(CONFIG_PATH)):
        input_ids = []
        attention_mask = []

        context = example.prompt
        input_ids = tokenizer.encode(context.strip(), add_special_tokens=False)
        explanation = example.completion
        added_ids = tokenizer.encode(explanation,
                                          add_special_tokens=False)
        encoder_input_tensor.append(input_ids)

        encoder_attention_mask_tensor.append([1] * len(input_ids[:args.max_enc_length]) + [0] * (args.max_enc_length - len(input_ids)))
        decoder_label_tensor.append(added_ids)
        print(f"input tensor {len(encoder_input_tensor)}")
        print(f"one item{len(input_ids)}")
    choices_input_ids = [ids[:args.max_enc_length] for ids in encoder_input_tensor]
    choices_input_ids = [ids + [tokenizer.pad_token_id] * (args.max_enc_length - len(ids)) for ids in
                         choices_input_ids]
    label_tensor = [ids[:args.max_dec_length] for ids in decoder_label_tensor]
    label_tensor = [ids + [tokenizer.pad_token_id] * (args.max_dec_length - len(ids))
                    for ids in label_tensor]
    encoder_input_tensor = torch.tensor(choices_input_ids, dtype=torch.long)
    encoder_attention_mask_tensor = torch.tensor(encoder_attention_mask_tensor, dtype=torch.long)
    decoder_label_tensor = torch.tensor(label_tensor, dtype=torch.long)
    print(decoder_label_tensor.shape)

    for f1, f2, f3 in zip(encoder_input_tensor[:2], encoder_attention_mask_tensor[:2], decoder_label_tensor[:2]):
        print("*** Example ***")
        if len(f1.shape) == 3:
            f1 = f1[0]
        for ids in f1:
            print("encoder input: %s" % tokenizer.decode(ids))
        # print("encoder attention mask: %s" % f2)
        print(f3)
        for ids in f3:
            if not ids == -100:
             print("decoder output: %s" % tokenizer.decode(ids))

    return TensorDataset(encoder_input_tensor, encoder_attention_mask_tensor, decoder_label_tensor)