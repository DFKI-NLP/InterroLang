import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.prediction.mistakes import show_mistakes_operation

conversation = CONVERSATION


def test_mistake_sample():
    """Test mistake sample functionality"""
    parse_text = ["mistake", "sample", "[E]"]

    return_s, status_code = show_mistakes_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/mistake/mistake_sample.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1


def test_mistake_count():
    """Test mistake count functionality"""
    parse_text = ["mistake", "count", "[E]"]

    return_s, status_code = show_mistakes_operation(conversation, parse_text, 0)

    file_html = open(f"./tests/html/mistake/mistake_count.html", "w")
    text = TEXT
    text += return_s
    text += "</body></html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
