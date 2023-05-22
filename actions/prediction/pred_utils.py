import json

import numpy as np
from logic.utils import read_precomputed_explanation_data


DD_MAPPING = {'dummy': 0, 'inform': 1, 'question': 2, 'directive': 3, 'commissive': 4}


def get_predictions_and_labels(name, indices):
    """
    Args:
        name: dataset name
        indices: indices of temp_dataset
    Returns:
        predictions and labels
    """
    json_list = read_precomputed_explanation_data(name)
    y_pred, y_true, ids = [], [], []

    if name == "daily_dialog":

        fileObject = open('./cache/daily_dialog/ig_explainer_daily_dialog_explanation.json', "r")
        jsonContent = fileObject.read()
        explanation_ls = json.loads(jsonContent)

        label2id = {'dummy': 0, 'inform': 1, 'question': 2, 'directive': 3, 'commissive': 4}

        for item in json_list:
            if item["batch"] in indices:
                y_pred.append(np.argmax(item["predictions"]))

        for item in explanation_ls:
            if item["index_running"] in indices:
                y_true.append(label2id[item["label"]])
        ids = np.array(indices)
    else:
        for item in json_list:
            if item["index_running"] in indices:
                y_pred.append(np.argmax(item["predictions"]))
                y_true.append(item["label"])
                ids.append(item["index_running"])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    id_array = np.array(ids)

    return y_pred, y_true, id_array
