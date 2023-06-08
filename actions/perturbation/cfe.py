from actions.perturbation.cfe_generation import CFEExplainer


def extract_id_cfe_number(parse_text):
    """

    Args:
        parse_text: parsed text from conversation

    Returns:
        id of text and number of cfe instances
    """
    num_list = []
    for item in parse_text:
        try:
            if int(item):
                num_list.append(int(item))
        except:
            pass
    if len(num_list) == 1:
        return num_list[0], 1
    elif len(num_list) == 2:
        return num_list[0], num_list[1]
    else:
        raise ValueError("Too many numbers in parse text!")


def get_text_by_id(conversation, _id):
    """

    Args:
        conversation: the current conversation
        _id: filtered id

    Returns:
        text from conversation with given id
    """
    texts = conversation.temp_dataset.contents['X']
    filtered_text = ''

    dataset_name = conversation.describe.get_dataset_name()

    if dataset_name == 'boolq':
        filtered_text += texts["question"][_id]
        filtered_text += " "
        filtered_text += texts["passage"][_id]
    elif dataset_name == 'daily_dialog':
        filtered_text += texts["dialog"][_id]
    else:
        filtered_text += texts["text"][_id]
    return filtered_text


def counterfactuals_operation(conversation, parse_text, i, **kwargs):
    """
    nlpcfe operation
    Args:
        conversation: the current conversation
        parse_text: parsed text from parser
        i: current counter
        **kwargs: additional args

    Returns:

    """

    _id, cfe_num = extract_id_cfe_number(parse_text)
    dataset_name = conversation.describe.get_dataset_name()

    instance = get_text_by_id(conversation, _id)

    cfe_explainer = CFEExplainer(dataset_name=dataset_name)
    same, diff = cfe_explainer.cfe(instance, cfe_num, _id=_id)

    if len(same) > 0:
        predicted_label = same[0][1]
    else:
        model = conversation.get_var("model").contents
        predicted_label = model(instance)

    return_s = ""

    return_s += "<ul>"
    return_s += '<li>'
    return_s += f"<b>[The original text]:</b> "
    return_s += f"{instance}"
    return_s += '</li>'

    if len(diff) > 0:
        flipped_label = diff[0][1]

        for i in range(len(diff)):
            return_s += '<li>'
            return_s += f"<b>[Counterfactual {i + 1}]:</b> "
            return_s += diff[i][2]
            return_s += '</li>'
        return_s += "</ul><br>"

        return_s += f"The predicted label <span style=\"background-color: #6CB4EE\">{predicted_label}</span> changes to <span style=\"background-color: #6CB4EE\">{flipped_label}</span>."

    else:
        return_s += f"This sentence is always classified as <b>{predicted_label}</b>!"

    return return_s, 1
