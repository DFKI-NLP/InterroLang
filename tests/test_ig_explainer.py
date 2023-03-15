import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from actions.explanation.feature_importance import FeatureAttributionExplainer
from explained_models.DataLoaderABC.hf_dataloader import HFDataloader
from explained_models.ModelABC.distilbert_qa_boolq import DistilbertQABoolModel
from explained_models.Tokenizer.tokenizer import HFTokenizer


def test_ig_explainer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "andi611/distilbert-base-uncased-qa-boolq"
    tokenizer = HFTokenizer(model_id)
    dataloader = HFDataloader(tokenizer=tokenizer.tokenizer, batch_size=1, number_of_instance=1)
    model = DistilbertQABoolModel(dataloader, num_labels=2, model_id=model_id)
    FeatureAttributionExplainer(model, device=device).generate_explanation()
