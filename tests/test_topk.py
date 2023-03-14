import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.Topk.topk import topk


def test_topk():
    k = 10

    # For whole folder testing
    # scores = topk("ig_explainer", k, data_path="./data/ig_explainer_boolq_explanation.json",
    #          res_path="./data/ig_explainer_boolq_attribution.json")

    # For single .py testing
    scores = topk("ig_explainer", k)

    assert len(scores) == k
