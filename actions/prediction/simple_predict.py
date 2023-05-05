"""Prediction operation."""
import string

def predict_operation(conversation, parse_text, i, max_num_preds_to_print=1, **kwargs):
    """The prediction operation."""
    model = conversation.get_var('model').contents
    parsed_id = ""
    if len(parse_text)>0:
        parsed_id = parse_text[i+1]
    while parsed_id[-1] in string.punctuation:
        parsed_id = parsed_id[:-1]
    if parsed_id.isdigit():
        id_val = int(parsed_id)
    else:
        return "Sorry, invalid id", 1

    dataset = conversation.get_var('dataset').contents["dataset_name"]
    if dataset=="boolq":
        instance = conversation.get_var('dataset').contents["X"].iloc[id_val]["passage"]
        question = conversation.get_var('dataset').contents["X"].iloc[id_val]["question"]
        instance_to_predict = {"question":question, "passage":instance}
    else:
        instance = conversation.get_var('dataset').contents["X"].iloc[id_val]["text"]
        instance_to_predict = instance
    prediction = model.predict_raw([instance_to_predict], dataset)[0]

    return_s = "Predicted class: "+conversation.get_var('dataset').contents["id2label"][prediction]
    return return_s, 1
