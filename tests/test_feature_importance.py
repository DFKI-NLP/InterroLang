import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.explanation.feature_importance import feature_importance_operation
from logic.dataset_description import DatasetDescription

from logic.conversation import Conversation

"""
pytest -q test_feature_importance.py           (in tests folder)

pytest -q test_feature_importance.py --global  (under root folder)
"""

conversation = Conversation(class_names={1: "True", 0: "False"})

datasetDescription = DatasetDescription(
    dataset_objective="predict to answer yes/no questions based on text passages",
    dataset_description="Boolean question answering (yes/no)",
    model_description="DistilBERT", name="boolq")

conversation.describe = datasetDescription


def test_feature_importance(for_test):
    """
    Test feature importance for a single instance with given id
    """

    parse_text = ["filter", "id", "33", "and", "nlpattribute", "topk", "1", "[E]"]

    return_s, status_code = feature_importance_operation(conversation, parse_text, 1, for_test)

    file_html = open("./html/feature_importance/feature_importance.html", "w")
    text = "<!DOCTYPE html><html><head></head><body>"
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()

    assert status_code == 1


def test_multiple_feature_importance(for_test):
    """
    Test feature importance for multiple instances with given ids
    """

    parse_text = ["filter", "id", "33", "or", "id", "151", "and", "nlpattribute", "topk", "2", "[E]"]

    return_s, status_code = feature_importance_operation(conversation, parse_text, 1, for_test)

    file_html = open("./html/feature_importance/multiple_feature_importance.html", "w")
    text = "<!DOCTYPE html><html><head></head><body>"
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()

    assert status_code == 1


def test_feature_importance_with_custom_input(for_test):
    """
    Test feature importance for custom input
    """

    parse_text = ["predict", "beginspan", "is", "a", "wolverine", "the", "same", "as", "a", "badger", "endspan",
                  "beginspan", "is", "this", "a", "good", "book", "endspan"]

    return_s, status_code = feature_importance_operation(conversation, parse_text, 1, for_test)

    file_html = open("./html/feature_importance/feature_importance_with_custom_input.html", "w")
    text = "<!DOCTYPE html><html><head></head><body>"
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()

    assert status_code == 1
