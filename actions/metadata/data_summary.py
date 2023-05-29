"""Data summary operation."""
import nltk
import pandas as pd

from actions.metadata.model import DATASET_TO_IDX

DATA_FLAG = ["train_data_name", "train_data_source", "train_data_language", "train_data_number", "test_data_name",
             "test_data_source", "test_data_language", "test_data_number"]

from timeout import timeout


def get_intro_text(flag, conversation):
    dataset_name = conversation.describe.get_dataset_name()
    df = pd.read_csv("./data/model_card.csv")

    idx = DATASET_TO_IDX[dataset_name]

    if flag in DATA_FLAG[:4]:
        text = "<b>Training Data Details: </b> <br><br>"
        text += "<table style='border: 1px solid black;'>"
        text += "<tr style='border: 1px solid black;'>"
        text += "<th> Name </th>"
        text += "<th> Content </th>"
        text += "</tr>"

        for i in DATA_FLAG[:4]:
            text += "<tr style='border: 1px solid black;'>"
            if i == flag:
                text += f"<td style='border: 1px solid black;'> <span style='background-color:yellow'>{i}</span> </td>"
            else:
                text += f"<td style='border: 1px solid black;'> {i} </td>"

            text += f"<td style='border: 1px solid black;'> {df[i][idx]} </td>"
            text += "</tr>"
        text += "</table><br>"
    else:
        text = "<b>Testing Data Details: </b> <br><br>"
        text += "<table style='border: 1px solid black;'>"
        text += "<tr style='border: 1px solid black;'>"
        text += "<th> Name </th>"
        text += "<th> Content </th>"
        text += "</tr>"

        for i in DATA_FLAG[4:]:
            text += "<tr style='border: 1px solid black;'>"
            if i == flag:
                text += f"<td style='border: 1px solid black;'> <span style='background-color:yellow'>{i}</span> </td>"
            else:
                text += f"<td style='border: 1px solid black;'> {i} </td>"

            text += f"<td style='border: 1px solid black;'> {df[i][idx]} </td>"
            text += "</tr>"
        text += "</table><br><br>"
    return text


@timeout(60)
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""

    flag = parse_text[i+1]

    if flag == '[e]':
        text = ''
    elif flag in DATA_FLAG:
        text = get_intro_text(flag, conversation)
    else:
        raise TypeError(f"The flag {flag} is not supported!")

    description = conversation.describe.get_dataset_description()
    text += f"The data contains information related to <b>{description}</b>.<br>"

    # List out the feature names
    f_names = list(conversation.temp_dataset.contents['X'].columns)

    f_string = "<ul>"
    for fn in f_names:
        f_string += f"<li>{fn}</li>"
    f_string += "</ul>"
    text += f"The exact <b>feature names</b> in the data are listed as follows:{f_string}<br>"

    class_list = list(conversation.class_names.values())
    text += "The dataset has following <b>labels</b>: "
    text += "<ul>"
    for i in range(len(class_list)):
        text += "<li>"
        text += str(class_list[i])
        text += "</li>"
    text += "</ul><br>"

    # Summarize performance
    dataset_name = conversation.describe.get_dataset_name()
    score = conversation.describe.get_eval_performance_for_hf_model(dataset_name, conversation.default_metric)

    # Note, if no eval data is specified this will return an empty string and nothing will happen.
    if score != "":
        text += score
        text += "<br><br>"

    return text, 1
