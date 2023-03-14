import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TransformerModel(Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def predict(self, data):
        """ Mirrors the sklearn predict function https://scikit-learn.org/stable/glossary.html#term-predict
        Arguments:
            data: Pandas DataFrame containing columns of text data
        """
        # Randomly generated vector of 0s and 1s.
        # TODO: Find a workaround for quick inference.
        return np.random.randint(2, size=len(data))

        # Actual code (below) takes ~4 minutes.
        predictions = []
        for i, instance in tqdm(data.iterrows(), total=len(data)):
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
