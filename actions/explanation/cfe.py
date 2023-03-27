from counterfactuals.cfe_generation_refactor import CFEExplainer, ALL_CTRL_CODES

model_id2label = {0: 'dummy', 1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}


def extract_id_cfe_number(parse_text):
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
    f_names = list(conversation.temp_dataset.contents['X'].columns)
    texts = conversation.temp_dataset.contents['X']
    filtered_text = ''
    for f in f_names:
        filtered_text += texts[f][_id]
        filtered_text += " "
    return filtered_text


def counterfactuals_operation(conversation, parse_text, i, **kwargs):
    # Parsed: filter id 54 and nlpcfe [E]

    _id, cfe_num = extract_id_cfe_number(parse_text)

    instance = get_text_by_id(conversation, _id)

    cfe_explainer = CFEExplainer()
    same, diff = cfe_explainer.cfe(instance, cfe_num, ctrl_code=ALL_CTRL_CODES, id2label=model_id2label)

    if len(same) > 0:
        predicted_label = same[0][1]
    else:
        model = conversation.get_var("model").contents
        predicted_label = model(instance)

    return_s = ""
    if len(diff) > 0:
        # [('oh , god , no thanks .', 'dummy'), ('oh , good boy , no thanks .', 'dummy')]
        return_s += "If you change the input as shown below, you will get a different class prediction. <br>"
        flipped_label = diff[0][1]
        return_s += f"The model will predict label <b>{flipped_label}</b>, " \
                    f"where the true label is <b>{predicted_label}</b>: <br>"

        return_s += "<ul>"
        for i in range(len(diff)):
            return_s += '<li>'
            return_s += diff[i][0]
            return_s += '</li>'
        return_s += "</ul>"
    else:
        return_s += f"This sentence is always classified as {predicted_label}!"