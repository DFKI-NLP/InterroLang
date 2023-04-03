"""Prediction operation."""
import numpy as np

from actions.utils import gen_parse_op_text, get_parse_filter_text


def handle_input(parse_text):
    num = None
    for item in parse_text:
        try:
            if int(item):
                num = int(item)
        except:
            pass
    return num


def predict_operation(conversation, parse_text, i, max_num_preds_to_print=1, **kwargs):
    """The prediction operation."""
    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    text = handle_input(parse_text)

    # Format return string
    return_s = ""

    if text is not None:
        model_predictions = model.predict(data, text)

        filter_string = gen_parse_op_text(conversation)

        if model_predictions.size == 1:
            return_s += f"The instance with <b>{filter_string}</b> is predicted "
            if conversation.class_names is None:
                prediction_class = str(model_predictions[0])
                return_s += f"<b>{prediction_class}</b>"
            else:
                class_text = conversation.class_names[model_predictions[0]]
                return_s += f"<b>{class_text}</b>."
        else:
            intro_text = get_parse_filter_text(conversation)
            return_s += f"{intro_text} the model predicts:"
            unique_preds = np.unique(model_predictions)
            return_s += "<ul>"
            for j, uniq_p in enumerate(unique_preds):
                return_s += "<li>"
                freq = np.sum(uniq_p == model_predictions) / len(model_predictions)
                round_freq = str(round(freq * 100, conversation.rounding_precision))

                if conversation.class_names is None:
                    return_s += f"<b>class {uniq_p}</b>, {round_freq}%"
                else:
                    class_text = conversation.class_names[uniq_p]
                    return_s += f"<b>{class_text}</b>, {round_freq}%"
                return_s += "</li>"
            return_s += "</ul>"
        return_s += "<br>"
    else:
        if parse_text[i+1] == "random":
            import random
            import time

            random.seed(time.time())
            f_names = list(data.columns)

            # Using random.randint doesn't work here somehow
            random_num = random.randint(0, len(data[f_names[0]]))
            filtered_text = ''

            dataset_name = conversation.describe.get_dataset_name()

            # Get the first column, also for boolq, we only need question column not passage
            if dataset_name == "boolq":
                for f in f_names[:2]:
                    filtered_text += data[f][random_num]
                    filtered_text += " "
            elif dataset_name == "daily_dialog":
                for f in f_names[:1]:
                    filtered_text += data[f][random_num]
                    filtered_text += " "
            elif dataset_name == "olid":
                pass
            else:
                raise NotImplementedError(f"The dataset {dataset_name} is not supported!")

            return_s += f"The random text is with <b>id {random_num}</b>: <br><br>"
            return_s += "<ul>"
            return_s += "<li>"
            return_s += f'The text is: {filtered_text}'
            return_s += "</li>"

            return_s += "<li>"
            model_predictions = model.predict(data, text)
            if conversation.class_names is None:
                prediction_class = str(model_predictions[0])
                return_s += f"The class name is not given, the prediction class is <b>{prediction_class}</b>"
            else:
                class_text = conversation.class_names[model_predictions[0]]
                return_s += f"The prediction is <b>{class_text}</b>."
            return_s += "</li>"
            return_s += "</ul>"
        else:
            raise NotImplementedError(f"The flag {parse_text[i+1]} is not supported!")
    return return_s, 1
