import numpy as np
from logic.utils import read_precomputed_explanation_data


MAPPING = {'dummy': 0, 'inform': 1, 'question': 2, 'directive': 3, 'commissive': 4}


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
        for i in range(len(json_list)):
            y_pred.append(json_list[i]["predictions"])
            y_true.append(MAPPING[json_list[i]["label"]])

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
