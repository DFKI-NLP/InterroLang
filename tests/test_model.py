import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_simulator import create_conversation, TEXT, CONVERSATION
from actions.metadata.model import model_operation

conversation = CONVERSATION


def test_model_description():
    """Test model description functionality"""
    parse_text = ["model", "[E]"]

    return_s, status_code = model_operation(conversation, parse_text, 1)

    file_html = open(f"./tests/html/model/model_description.html", "w")
    text = TEXT
    text += return_s
    text += "</html>"
    file_html.write(text)

    # Saving the data into the HTML file
    file_html.close()
    assert status_code == 1
