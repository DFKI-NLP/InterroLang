"""The app main."""
import gin
import json
import logging
import os
import traceback

from flask import Flask
from flask import render_template, request, Blueprint
from logging.config import dictConfig

from logic.core import ExplainBot
from logic.sample_prompts_by_action import sample_prompt_for_action


# gunicorn doesn't have command line flags, using a gin file to pass command line args
@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl


# Parse gin global config
gin.parse_config_file("global_config.gin")

# Get args
args = GlobalArgs()

bp = Blueprint('host', __name__, template_folder='templates')

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse application level configs
gin.parse_config_file(args.config)

# Setup the explainbot
BOT = ExplainBot()


@bp.route('/')
def home():
    """Load the explanation interface."""
    app.logger.info("Loaded Login")
    objective = BOT.conversation.describe.get_dataset_objective()

    BOT.conversation.build_temp_dataset()

    dataset = BOT.conversation.describe.get_dataset_name()

    return render_template("index.html", currentUserId="user", datasetObjective=objective, dataset=dataset)


@bp.route("/log_feedback", methods=['POST'])
def log_feedback():
    """Logs feedback"""
    feedback = request.data.decode("utf-8")
    app.logger.info(feedback)
    split_feedback = feedback.split(" || ")

    message = f"Feedback formatted improperly. Got: {split_feedback}"
    assert split_feedback[0].startswith("MessageID: "), message
    assert split_feedback[1].startswith("Feedback: "), message
    assert split_feedback[2].startswith("Username: "), message

    message_id = split_feedback[0][len("MessageID: "):]
    feedback_text = split_feedback[1][len("Feedback: "):]
    username = split_feedback[2][len("Username: "):]

    logging_info = {
        "id": message_id,
        "feedback_text": feedback_text,
        "username": username
    }

    BOT.log(logging_info)
    return ""


@bp.route("/sample_prompt", methods=["Post"])
def sample_prompt():
    """Samples a prompt"""
    data = json.loads(request.data)
    action = data["action"]
    username = data["thisUserName"]

    prompt = sample_prompt_for_action(action,
                                      BOT.prompts.filename_to_prompt_id,
                                      BOT.prompts.final_prompt_set,
                                      real_ids=BOT.conversation.get_training_data_ids())

    logging_info = {
        "username": username,
        "requested_action_generation": action,
        "generated_prompt": prompt
    }
    BOT.log(logging_info)

    return prompt


@bp.route("/get_response", methods=['POST'])
def get_bot_response():
    """Load the box response."""
    if request.method == "POST":

        try:
            data = json.loads(request.data)
            if data['custom_input'] == '0':
                app.logger.info("generating the bot response")
                user_text = data["userInput"]
                conversation = BOT.conversation
                response = BOT.update_state(user_text, conversation)
            else:
                user_text = data["userInput"]
                BOT.conversation.custom_input = user_text
                BOT.conversation.used = False
                app.logger.info(f"[CUSTOM INPUT] {user_text}")
                response = "You have given a custom input. Please enter a follow-up question or prompt!" + "<>" + "Entered custom input: " + user_text
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
        return response


@bp.route("/custom_input", methods=["Post"])
def custom_input():
    data = json.loads(request.data)
    custom_input = data["custom_input"]
    username = data["thisUserName"]

    BOT.conversation.custom_input = custom_input
    BOT.conversation.used = False

    app.logger.info("custom_input: " + custom_input)

    return custom_input


@bp.route("/filter_dataset", methods=["POST"])
def filter_dataset():
    filter_text = json.loads(request.data)["filterMsgText"]
    df = BOT.conversation.stored_vars["dataset"].contents["X"]
    if len(filter_text) > 2:
        filtered_df = df[df[BOT.text_fields].apply(lambda row: row.str.contains(filter_text)).any(axis=1)]

        BOT.conversation.temp_dataset.contents["X"] = filtered_df
        app.logger.info(f"{len(filtered_df)} instances of {BOT.conversation.describe.dataset_name} include the filter string '{filter_text}'")
        final_df = filtered_df
    else:
        final_df = df
    return {
        'jsonData': final_df.to_json(orient="index"),
        'totalDataLen': len(df)
    }


@bp.route("/reset_temp_dataset", methods=["Post"])
def reset_temp_dataset():
    data = json.loads(request.data)
    username = data["thisUserName"]

    # Reset the tempdataset
    BOT.conversation.build_temp_dataset()

    app.logger.info("Reset temp dataset successfully!")

    return "reset temp_dataset"


app = Flask(__name__)
app.register_blueprint(bp, url_prefix=args.baseurl)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    # clean up storage file on restart
    app.logger.info(f"Launching app from config: {args.config}")
    app.run(debug=False, port=4455, host='0.0.0.0')
