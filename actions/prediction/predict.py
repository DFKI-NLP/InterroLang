"""Prediction operation."""
import json
import os
import csv
import random
import time
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from actions.util_functions import gen_parse_op_text, get_parse_filter_text


def handle_input(parse_text):
    num = None
    for item in parse_text:
        try:
            if int(item):
                num = int(item)
        except:
            pass
    return num


def store_results(inputs, predictions, cache_path):
    """
    Store custom inputs and its predictions in csv file
    Args:
        inputs: custom input
        predictions: corresponding predictions
        cache_path: path to cache/csv
    """
    if not os.path.exists(cache_path):
        with open(cache_path, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write header
            writer.writerow(["idx", "Input text", "Prediction"])
            for i in range(len(inputs)):
                writer.writerow([i, inputs[i], predictions[i]])
            file.close()

    else:
        rows = []
        with open(cache_path, 'r', ) as file:
            fieldnames = ["idx", "Input text", "Prediction"]
            reader = csv.DictReader(file, fieldnames=fieldnames)

            for row in reader:
                rows.append(row)
            file.close()
        length = len(rows)

        with open(cache_path, 'w', newline='') as file:
            fieldnames = ["idx", "Input text", "Prediction"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(1, length):
                writer.writerow(rows[i])

            for i in range(len(inputs)):
                writer.writerow({
                    "idx": i + length - 1,
                    "Input text": inputs[i],
                    "Prediction": predictions[i]
                })
            file.close()


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

    cache_path = f"./cache/{dataset_name}/{dataset_name}_custom_input.csv"

    if dataset_name == "boolq":
        model = AutoModelForSequenceClassification.from_pretrained("andi611/distilbert-base-uncased-qa-boolq",
                                                                   num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained("andi611/distilbert-base-uncased-qa-boolq")

        if os.path.exists(cache_path):
            with open(cache_path, 'r') as file:
                fieldnames = ["idx", "Input text", "Prediction"]
                reader = csv.DictReader(file, fieldnames=fieldnames)

                for string in inputs:
                    flag = False
                    for row in reader:
                        if row["Input text"] == string:
                            predictions.append(int(row["Prediction"]))
                            flag = True
                            break
                    if flag:
                        continue

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
        else:
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

    store_results(inputs, predictions, cache_path)

    return return_s


def random_prediction(model, data, conversation, text):
    """randomly pick an instance from the dataset and make the prediction"""
    return_s = ''

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
        for f in f_names[:1]:
            filtered_text += data[f][random_num]
            filtered_text += " "
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not supported!")

    return_s += f"The random text is with <b>id {random_num}</b>: <br><br>"
    return_s += "<ul>"
    return_s += "<li>"
    return_s += f'The text is: {filtered_text}'
    return_s += "</li>"

    return_s += "<li>"
    if conversation.describe.get_dataset_name() != "daily_dialog":
        model_predictions = model.predict(data, text)
    else:
        model_predictions = get_prediction_by_id_da(random_num)
    if conversation.class_names is None:
        prediction_class = str(model_predictions[0])
        return_s += f"The class name is not given, the prediction class is <b>{prediction_class}</b>"
    else:
        class_text = conversation.class_names[model_predictions[0]]
        return_s += f"The prediction is <b>{class_text}</b>."
    return_s += "</li>"
    return_s += "</ul>"

    return return_s


def get_prediction_by_id_da(_id):
    name = 'daily_dialog'
    data_path = f"./cache/{name}/ig_explainer_{name}_prediction.json"
    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    return np.array([np.argmax(json_list[_id])])


def prediction_with_id(model, data, conversation, text):
    """Get the prediction of an instance with ID"""
    return_s = ''
    if conversation.describe.get_dataset_name() != 'daily_dialog':
        model_predictions = model.predict(data, text)
    else:
        model_predictions = get_prediction_by_id_da(text)

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
    # "beginspan", "is", "this", "a", "good", "book", "endspan"]

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
