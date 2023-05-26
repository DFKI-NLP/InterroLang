"""Describes the model."""

from timeout import timeout

import pandas as pd

DATASET_TO_IDX = {"boolq": 1, "olid":0, "daily_dialog": 2}



@timeout(60)
def model_operation(conversation, parse_text, i, **kwargs):
    """Model description."""

    model_flag = ["model_name", "model_summary", "task", "epochs", "lr", "loss", "optimizer"]
    flag = parse_text[i+1]
    if flag == '[e]':
        objective = conversation.describe.get_dataset_objective()
        model = conversation.describe.get_model_description()
        text = f"I use a <em>{model}</em> model to {objective}.<br><br>"
    elif flag in model_flag:
        dataset_name = conversation.describe.get_dataset_name()
        df = pd.read_csv("./data/model_card.csv")

        idx = DATASET_TO_IDX[dataset_name]

        if flag in model_flag[3:]:
            text = "<b>Implementation Details: </b> <br><br>"
            text += "<table style='border: 1px solid black;'>"
            text += "<tr style='border: 1px solid black;'>"
            text += "<th> Name </th>"
            text += "<th> Content </th>"
            text += "</tr>"

            for i in model_flag[3:]:
                text += "<tr style='border: 1px solid black;'>"
                if i == flag:
                    text += f"<td style='border: 1px solid black;'> <span style='background-color:yellow'>{i}</span> </td>"
                else:
                    text += f"<td style='border: 1px solid black;'> {i} </td>"

                text += f"<td style='border: 1px solid black;'> {df[i][idx]} </td>"
                text += "</tr>"
            text += "</table><br>"
        else:
            text = "<b>General Information: </b> <br><br>"
            text += "<table style='border: 1px solid black;'>"
            text += "<tr style='border: 1px solid black;'>"
            text += "<th> Name </th>"
            text += "<th> Content </th>"
            text += "</tr>"

            for i in model_flag[:3]:
                text += "<tr style='border: 1px solid black;'>"
                if i == flag:
                    text += f"<td style='border: 1px solid black;'> <span style='background-color:yellow'>{i}</span> </td>"
                else:
                    text += f"<td style='border: 1px solid black;'> {i} </td>"

                text += f"<td style='border: 1px solid black;'> {df[i][idx]} </td>"
                text += "</tr>"
            text += "</table><br>"

    else:
        raise TypeError(f"Model flag {parse_text[i+1]} is not supported!")
    return text, 1
