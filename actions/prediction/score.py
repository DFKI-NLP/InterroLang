"""Score operation.

This operation computes a score metric on the data or the eval data.
"""
import json

import numpy as np

from actions.util_functions import gen_parse_op_text


def get_predictions_and_labels(dataset_name):
    """

    Args:
        dataset_name: The name of dataset

    Returns:
        Arrays of predictions and actual labels
    """
    data_path = f"./cache/{dataset_name}/ig_explainer_{dataset_name}_explanation.json"
    fileObject = open(data_path, "r")
    jsonContent = fileObject.read()
    json_list = json.loads(jsonContent)

    predictions = []
    labels = []

    for item in json_list:
        labels.append(item["label"])
        predictions.append(np.argmax(item["predictions"]))

    y_true = np.array(labels)
    y_pred = np.array(predictions)
    return y_true, y_pred


def score_operation(conversation, parse_text, i, **kwargs):
    """Self description."""

    # Get the name of the metric
    metric = parse_text[i+1]

    if metric == "default":
        metric = conversation.default_metric

    # model = conversation.get_var('model').contents
    #
    # data = conversation.temp_dataset.contents['X']

    # Get the dataset name
    dataset_name = conversation.describe.get_dataset_name()

    y_true, y_pred = get_predictions_and_labels(dataset_name)

    filter_string = gen_parse_op_text(conversation)
    if len(filter_string) <= 0:
        data_name = "the <b>all</b> the data"
    else:
        data_name = f"the data where <b>{filter_string}</b>"
    text = conversation.describe.get_score_text(y_true,
                                                y_pred,
                                                metric,
                                                conversation.rounding_precision,
                                                data_name)

    text += "<br><br>"
    return text, 1
