import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.output_test import create_conversation, TEXT, CONVERSATION
from actions.about.self import self_operation

conversation = CONVERSATION


def test_whatami():
    """Test whatami functionality"""
    parse_text = ["self", "[E]"]

    return_s, status_code = self_operation(conversation, parse_text, 1)

    file_html = open(f"./tests/html/self/whatami.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
