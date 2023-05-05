import json

import numpy as np
import torch
from torch.nn import Module
#from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TransformerModel(Module):
    def __init__(self, model_id, name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dataset_name = name

    def __call__(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def predict(self, data, text):
        """ Mirrors the sklearn predict function https://scikit-learn.org/stable/glossary.html#term-predict
        Arguments:
            data: Pandas DataFrame containing columns of text data
            text: preprocessed parse_text
        """
        path = f"./cache/{self.dataset_name}/ig_explainer_{self.dataset_name}_explanation.json"
        try:
            fileObject = open(path, "r")
            jsonContent = fileObject.read()
            json_list = json.loads(jsonContent)
        except:
            raise FileNotFoundError(f"The required cache with path {path} doesn't exist!")

        if text is None:
            temp = []
            for item in json_list:
                temp.append(item["label"])

            return np.array(temp)
        else:
            res = list([json_list[text]["label"]])
            return np.array(res)

        # # Randomly generated vector of 0s and 1s.
        # return np.random.randint(2, size=len(data))
        #
        # # Actual code (below) takes ~4 minutes.
        # predictions = []
        # for i, instance in tqdm(data.iterrows(), total=len(data)):
        #     encodings = self.tokenizer(
        #         instance['question'],  # TODO: Automatically get info about columns and make dynamic
        #         instance['passage'],
        #         padding=True,
        #         truncation=True,
        #         return_tensors='pt'
        #     )
        #     out = self.model(**encodings)
        #     pred = int(torch.argmax(out.logits, axis=1))
        #     predictions.append(pred)
        # return np.array(predictions)

    def predict_raw(self, data, dataset):
        """ Mirrors the sklearn predict function https://scikit-learn.org/stable/glossary.html#term-predict
        Arguments:
            data: list of strings
            dataset: string indicating which dataset is used
        """
        predictions = []
        for i, instance in enumerate(data):
            if dataset=="boolq":
                encodings = self.tokenizer(
                    instance['question'],  # TODO: Automatically get info about columns and make dynamic
                    instance['passage'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                out = self.model(**encodings)
                pred = int(torch.argmax(out.logits, axis=1))
                predictions.append(pred)
        return np.array(predictions)
