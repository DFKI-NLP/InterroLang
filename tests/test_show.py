import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.output_test import create_conversation, TEXT, CONVERSATION
from actions.metadata.show_data import show_operation

conversation = CONVERSATION


def test_show_data(root_path):
    """Test whatami functionality"""
    parse_text = ["show", "[E]"]

    return_s, status_code = show_operation(conversation, parse_text, 1)

    file_html = open(f"{root_path}/html/show/show_data.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
