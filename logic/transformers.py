import json
import numpy as np
from torch.nn import Module
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

        # Get indices of dataset to filter json_list with
        data_indices = data.index.to_list()

        if text is None:
            temp = []
            for item in json_list:
                if item["index_running"] in data_indices:
                    temp.append(item["label"])

            return np.array(temp)
        else:
            res = list([json_list[text]["label"]])
            return np.array(res)
