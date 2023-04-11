import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.output_test import create_conversation, TEXT, CONVERSATION
from actions.metadata.data_summary import data_operation

conversation = CONVERSATION


def test_data_summary(root_path):
    """Test whatami functionality"""
    parse_text = ["data", "[E]"]

    return_s, status_code = data_operation(conversation, parse_text, 1)

    file_html = open(f"{root_path}/html/data/data_summary.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
