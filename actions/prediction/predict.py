"""Prediction operation."""
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def prediction_with_custom_input(parse_text, conversation):
    """
    Predict the custom input from user that is not contained in the dataset
    Args:
        parse_text: parsed text from parse
        conversation: Conversation object

    Returns:
        format string with inputs and predictions
    """
    # beginspan token token ... token endspan
    begin_idx = [i for i, x in enumerate(parse_text) if x == 'beginspan']
    end_idx = [i for i, x in enumerate(parse_text) if x == 'endspan']

    if begin_idx == [] or end_idx == []:
        return None

    if len(begin_idx) != len(end_idx):
        return None

    inputs = []
    for i in list(zip(begin_idx, end_idx)):
        temp = " ".join(parse_text[i[0] + 1: i[1]])

        if temp != '':
            inputs.append(temp)

    if len(inputs) == 0:
        return None

    predictions = []

    dataset_name = conversation.describe.get_dataset_name()
    if dataset_name == "boolq":
        model = AutoModelForSequenceClassification.from_pretrained("andi611/distilbert-base-uncased-qa-boolq",
                                                                   num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")

        for string in inputs:
            encoding = tokenizer.encode_plus(string, return_tensors='pt')
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
            }
            output_model = model(**input_model)[0]

            # Get logit
            output_model = np.argmax(output_model.cpu().detach().numpy())
            predictions.append(output_model)

    elif dataset_name == "daily_dialog":
        pass
    elif dataset_name == "olid":
        pass
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not supported!")

    class_name = conversation.class_names

    return_s = ""
    if class_name is not None:
        for i in range(len(predictions)):
            return_s += "<ul>"
            return_s += "<li>"
            return_s += f"Your input is: <b>{inputs[i]}</b>"
            return_s += "</li>"

            return_s += "<li>"
            return_s += f"The prediction is: <b>{class_name[predictions[i]]}</b>"
            return_s += "</li>"
            return_s += "</ul><br><br>"
    else:
        return_s += "<ul>"
        return_s += "<li>"
        return_s += f"Your input is: {inputs[i]}"
        return_s += "</li>"

        return_s += "<li>"
        return_s += f"The prediction is: {predictions[i]}"
        return_s += "</li>"
        return_s += "</ul>"

    return return_s


def random_prediction(model, data, conversation, text):
    """randomly pick an instance from the dataset and make the prediction"""
    return_s = ''
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

    return return_s


def prediction_with_id(model, data, conversation, text):
    """Get the prediction of an instance with ID"""
    return_s = ''
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

    return return_s


def predict_operation(conversation, parse_text, i, **kwargs):
    """The prediction operation."""
    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X']

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    # For testing custom input
    # parse_text = ["predict", "beginspan", "is", "a", "wolverine", "the", "same", "as", "a", "badger", "endspan",
    # "beginspan", "is", "a", "wolverine", "the", "same", "as", "a", "badger", "endspan"]

    predictions = prediction_with_custom_input(parse_text, conversation)
    if predictions is not None:
        return predictions, 1

    text = handle_input(parse_text)

    if text is not None:
        return_s = prediction_with_id(model, data, conversation, text)
    else:
        if parse_text[i + 1] == "random":
            return_s = random_prediction(model, data, conversation, text)
        else:
            raise NotImplementedError(f"The flag {parse_text[i+1]} is not supported!")
    return return_s, 1
