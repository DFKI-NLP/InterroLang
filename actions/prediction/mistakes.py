"""Show model mistakes"""
import json

import gin
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from actions.utils import get_parse_filter_text, get_rules


def one_mistake(y_true, y_pred, conversation, intro_text):
    """One mistake text"""
    label = y_true[0]
    prediction = y_pred[0]

    label_text = conversation.get_class_name_from_label(label)
    predict_text = conversation.get_class_name_from_label(prediction)

    if label == prediction:
        correct_text = "correct"
    else:
        correct_text = "incorrect"

    return_string = (f"{intro_text} the model predicts <em>{predict_text}</em> and the ground"
                     f" label is <em>{label_text}</em>, so the model is <b>{correct_text}</b>!")
    return return_string


def sample_mistakes(y_true, y_pred, conversation, intro_text, ids):
    """Sample mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_num = np.sum(y_true != y_pred)
        total_num = len(y_true)
        incorrect_data = ids[y_true != y_pred]

        error_rate = round(incorrect_num / total_num, conversation.rounding_precision)
        return_string = (f"{intro_text} the model is incorrect {incorrect_num} out of {total_num} "
                         f"times (error rate {error_rate}). Here are the ids of instances the model"
                         f" predicts incorrectly:<br><br>{incorrect_data}")

    return return_string


def train_tree(data, target, depth: int = 1):
    """Trains a decision tree"""
    dt_string = []
    tries = 0
    while len(dt_string) < 3 and tries < 10:
        tries += 1
        dt = DecisionTreeClassifier(max_depth=depth).fit(data, target)
        dt_string = get_rules(dt,
                              feature_names=list(data.columns),
                              class_names=["correct", "incorrect"])
        depth += 1

    return dt_string


def typical_mistakes(data, y_true, y_pred, conversation, intro_text, ids):
    """Typical mistakes sub-operation"""
    if len(y_true) == 1:
        return_string = one_mistake(y_true, y_pred, conversation, intro_text)
    else:
        incorrect_vals = y_true != y_pred
        return_options = train_tree(data, incorrect_vals)

        if len(return_options) == 0:
            return "I couldn't find any patterns for mistakes the model typically makes."

        return_string = f"{intro_text} the model typically predicts incorrect:<br><br>"
        for rule in return_options:
            return_string += rule + "<br><br>"

    return return_string


def get_predictions_and_labels(name):
    """
    Args:
        name: dataset name
    Returns:
        predictions and labels
    """
    data_path = f"./cache/{name}/ig_explainer_{name}_explanation.json"
    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)
    y_pred, y_true = [], []

    for item in json_list:
        y_pred.append(np.argmax(item["predictions"]))
        y_true.append(item["label"])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return y_pred, y_true, len(json_list)


@gin.configurable
def show_mistakes_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows the model mistakes."""

    # Get dataset name
    name = conversation.describe.get_dataset_name()
    y_pred, y_true, length = get_predictions_and_labels(name)
    ids = np.array([i for i in range(length)])

    # The filtering text
    intro_text = get_parse_filter_text(conversation)

    if len(y_true) == 0:
        return "There are no instances in the data that meet this description.<br><br>", 0

    if np.sum(y_true == y_pred) == len(y_true):
        if len(y_true) == 1:
            return f"{intro_text} the model predicts correctly!<br><br>", 1
        else:
            return f"{intro_text} the model predicts correctly on all the instances in the data!<br><br>", 1

    if parse_text[i+1] == "sample":
        return_string = sample_mistakes(y_true,
                                        y_pred,
                                        conversation,
                                        intro_text,
                                        ids)
    # elif parse_text[i+1] == "typical":
    #     return_string = typical_mistakes(data,
    #                                      y_true,
    #                                      y_pred,
    #                                      conversation,
    #                                      intro_text,
    #                                      ids)
    else:
        raise NotImplementedError(f"No mistake type {parse_text[i+1]}")

    return_string += "<br><br>"
    return return_string, 1
