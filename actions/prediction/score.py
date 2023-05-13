"""Score operation.

This operation computes a score metric on the data or the eval data.
"""
import json

import numpy as np

from actions.util_functions import gen_parse_op_text

MAPPING = {'dummy': 0, 'inform': 1, 'question': 2, 'directive': 3, 'commissive': 4}


def get_predictions_and_labels(dataset_name):
    """

    Args:
        dataset_name: The name of dataset

    Returns:
        Arrays of predictions and actual labels
    """
    data_path = f"./cache/{dataset_name}/ig_explainer_{dataset_name}_explanation.json"
    if dataset_name == 'daily_dialog':
        pred_path = f"./cache/{dataset_name}/ig_explainer_{dataset_name}_prediction.json"
        fileObject = open(pred_path, "r")
        jsonContent = fileObject.read()
        pred_list = json.loads(jsonContent)

    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    predictions = []
    labels = []

    if dataset_name == 'daily_dialog':
        for i in range(len(json_list)):
            labels.append(MAPPING[json_list[i]["label"]])
            predictions.append(pred_list[i]["predictions"])
    else:
        for item in json_list:
            labels.append(item["label"])
            predictions.append(np.argmax(item["predictions"]))

    y_true = np.array(labels)
    y_pred = np.array(predictions)
    return y_true, y_pred


def score_operation(conversation, parse_text, i, **kwargs):
    """Self description."""

    # Get the name of the metric
    metric = parse_text[i + 1]

    # Get the dataset name
    dataset_name = conversation.describe.get_dataset_name()

    average = None
    if dataset_name == "daily_dialog":
        flags = ["micro", "macro", "weighted"]
        try:
            average = parse_text[i + 2]
        except ValueError:
            pass
        if metric not in ["default", "accuracy", "roc"]:
            if average not in flags:
                raise NotImplementedError(f"Flag {average} is not supported!")

    y_true, y_pred = get_predictions_and_labels(dataset_name)

    if metric == "default" or metric == 'accuracy':
        metric = conversation.default_metric
        if dataset_name == 'daily_dialog':
            y_pred = np.argmax(y_pred, axis=1)

    filter_string = gen_parse_op_text(conversation)
    if len(filter_string) <= 0:
        data_name = "the <b>all</b> the data"
    else:
        data_name = f"the data where <b>{filter_string}</b>"
    multi_class = True if dataset_name == 'daily_dialog' else False
    text = conversation.describe.get_score_text(y_true,
                                                y_pred,
                                                metric,
                                                conversation.rounding_precision,
                                                data_name,
                                                multi_class,
                                                average)

    text += "<br><br>"
    return text, 1
