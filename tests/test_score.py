import sys
import os



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.prediction.score import score_operation

conversation = CONVERSATION


def test_score_accuracy():
    """Test score accuracy functionality"""
    parse_text = ["score", "accuracy", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/score_accuracy.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_default():
    """Test score default functionality"""
    parse_text = ["score", "default", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/score_default.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_f1():
    """Test score f1 functionality"""
    parse_text = ["score", "f1", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/score_f1.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_precision():
    """Test score precision functionality"""
    parse_text = ["score", "precision", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/score_precision.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_recall():
    """Test score recall functionality"""
    parse_text = ["score", "recall", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/score_recall.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_score_roc():
    """Test score roc functionality"""
    parse_text = ["score", "roc", "[E]"]

    return_s, status_code = score_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/score/score_roc.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
