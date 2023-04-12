import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.explanation.feature_importance import feature_importance_operation

"""
pytest -q test_feature_importance.py           (in tests folder)

pytest -q test_feature_importance.py --global  (under root folder)
"""

conversation = CONVERSATION


def test_feature_importance():
    """
    Test feature importance for a single instance with given id
    """

    parse_text = ["filter", "id", "33", "and", "nlpattribute", "topk", "1", "[E]"]

    return_s, status_code = feature_importance_operation(conversation, parse_text, 1)

    file_html = open(f"./tests/html/feature_importance/feature_importance.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()

    assert status_code == 1


def test_multiple_feature_importance():
    """
    Test feature importance for multiple instances with given ids
    """

    parse_text = ["filter", "id", "33", "or", "id", "151", "and", "nlpattribute", "topk", "2", "[E]"]

    return_s, status_code = feature_importance_operation(conversation, parse_text, 1)

    file_html = open(f"./tests/html/feature_importance/multiple_feature_importance.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()

    assert status_code == 1


def test_feature_importance_with_custom_input():
    """
    Test feature importance for custom input
    """

    parse_text = ["predict", "beginspan", "is", "a", "wolverine", "the", "same", "as", "a", "badger", "endspan",
                  "beginspan", "is", "this", "a", "good", "book", "endspan"]

    return_s, status_code = feature_importance_operation(conversation, parse_text, 1)

    file_html = open(f"./tests/html/feature_importance/feature_importance_with_custom_input.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()

    assert status_code == 1


def test_feature_importance_all():
    """
    Test feature importance for a single instance with given id
    """

    parse_text = ["filter", "id", "53", "and", "nlpattribute", "all", "[E]"]

    return_s, status_code = feature_importance_operation(conversation, parse_text, 1)

    file_html = open(f"./tests/html/feature_importance/feature_importance_all.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()

    assert status_code == 1
