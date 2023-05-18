from actions.perturbation.cfe_generation import CFEExplainer, ALL_CTRL_CODES


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
    f_names = list(conversation.temp_dataset.contents['X'].columns)
    texts = conversation.temp_dataset.contents['X']
    filtered_text = ''

    # Get the first column, also for boolq, we only need question column not passage
    for f in f_names[:1]:
        filtered_text += texts[f][_id]
        filtered_text += " "
    return filtered_text


def get_text_by_id_from_csv(_id):
    """
    Args:
        _id: filtered id

    Returns:
        text from csv file with given id
    """
    import pandas as pd

    df = pd.read_csv('./data/da_test_set_with_indices.csv')
    text = df["dialog"][_id]
    label = df["act"][_id]
    return text, label


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

    # Needed for polyjuice
    # import nltk
    # nltk.download('omw-1.4')
    #
    # import spacy
    # spacy.load('en_core_web_sm')

    _id, cfe_num = extract_id_cfe_number(parse_text)
    dataset_name = conversation.describe.get_dataset_name()

    if dataset_name == "boolq":
        instance = get_text_by_id(conversation, _id)
    elif dataset_name == 'daily_dialog':
        instance, label = get_text_by_id_from_csv(_id)
    else:
        instance = get_text_by_id(conversation, _id)

    cfe_explainer = CFEExplainer(dataset_name=dataset_name)
    same, diff = cfe_explainer.cfe(instance, cfe_num, ctrl_code=ALL_CTRL_CODES, _id=_id)

    if len(same) > 0:
        predicted_label = same[0][1]
    else:
        model = conversation.get_var("model").contents
        predicted_label = model(instance)

    return_s = ""
    if len(diff) > 0:
        return_s += "<ul>"
        return_s += '<li>'
        return_s += f"<b>[The original text]:</b> "
        return_s += f"{instance}"
        return_s += '</li>'

        flipped_label = diff[0][1]

        for i in range(len(diff)):
            return_s += '<li>'
            return_s += f"<b>[Counterfactual {i+1}]:</b> "
            return_s += diff[i][0]
            return_s += '</li>'
        return_s += "</ul><br>"

        return_s += f"the predicted label <b>{predicted_label}</b> changes to <b>{flipped_label}</b>. <br>"

    else:
        return_s += f"This sentence is always classified as <b>{predicted_label}</b>!"

    return return_s, 1
