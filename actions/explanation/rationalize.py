import pandas as pd

from actions.prediction.predict import prediction_with_custom_input
from timeout import timeout
import json
import pandas as pd


def get_results(dataset_name, data_path):
    """
    Get the rationlize result

    Args:
        data_path: path to json file
    Returns:
        results: results in csv format
    """
    path = data_path + dataset_name + "/dolly-rationales.csv"
    results = pd.read_csv(path)

    return results


def get_few_shot_str(csv_filename, num_shots=3):
    few_shot_str = ""
    gpt_rationales = pd.read_csv(csv_filename).sample(frac=1).reset_index()
    for i, row in gpt_rationales.iterrows():
        few_shot_str += row["prompt"] + row["completion"] + "\n"
        if i == num_shots - 1:
            break
    return few_shot_str


def formalize_output(dataset_name, text):
    return_s = ""
    if dataset_name == "boolq":
        return_s += "<b>"
        return_s += text[0: 8]
        return_s += "</b>"

        idx_p = text.index("Passage")

        return_s += text[8: idx_p]
        return_s += "<br>"
        return_s += "<b>"
        return_s += text[idx_p: idx_p + 8]
        return_s += "</b>"
        return_s += text[idx_p + 8:]
    elif dataset_name == "daily_dialog":
        return_s += "<b>"
        return_s += text[0: 7]
        return_s += "</b>"
        return_s += text[7:]
    else:
        return_s += "<b>"
        return_s += text[0: 6]
        return_s += "</b>"
        return_s += text[6:]
    return return_s


def get_few_shot_result(few_shot_str, text, intro, instruction, conversation, pred_str):
    return_s = ""
    prompt = f"{few_shot_str}" \
             f"{text}\n" \
             f"{intro}\n" \
             f"{instruction}\n"
    print(f"[rationalize operation]\n=== PROMPT ===\n{prompt}")

    input_ids = conversation.decoder.gpt_tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device="cpu")
    generation = conversation.decoder.gpt_model.generate(
        input_ids,
        max_length=2048,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.7
    )
    decoded_generation = conversation.decoder.gpt_tokenizer.decode(generation[0], skip_special_tokens=True)

    explanation = decoded_generation.split(instruction)[1]

    return_s += "<b>Original text:</b> " + text \
                + "<br><b>Prediction:</b> " + pred_str \
                + "<br><b>Explanation:</b> " + explanation

    return return_s


# @timeout(60)
def rationalize_operation(conversation, parse_text, i, simulation, data_path="./cache/", **kwargs):
    dataset_name = conversation.describe.get_dataset_name()

    if dataset_name == "boolq":
        gpt_rationales = "cache/boolq/GPT-3.5_rationales_BoolQ_val_400.csv"
    elif dataset_name == "olid":
        gpt_rationales = "cache/olid/GPT-4_rationales_OLID_val_132.csv"
    elif dataset_name == "daily_dialog":
        gpt_rationales = "cache/daily_dialog/GPT-4_rationales_DD_test_200.csv"
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported!")

    if conversation.custom_input is not None and conversation.used is False:
        print("rationale")
        few_shot_str = get_few_shot_str(gpt_rationales)

        res = prediction_with_custom_input(conversation)
        df = pd.read_csv(f"./cache/{dataset_name}/{dataset_name}_custom_input.csv")
        prediction = df["Prediction"][df['Prediction'].index.to_list()[-1]]
        print(prediction)
        pred_str = conversation.class_names[prediction]

        if dataset_name == "boolq":
            intro = f"Answer: {pred_str}"
            instruction = "Please explain the answer: "
        elif dataset_name == "olid":
            other_class_names = ", ".join(
                [conversation.class_names[c] for c in conversation.class_names if conversation.class_names[c] not in [pred_str, "dummy"]])
            intro = f"The dialogue act of this text has been classified as {pred_str} (over {other_class_names})."
            instruction = "Please explain why: "
        elif dataset_name == "daily_dialog":
            intro = f"The tweet has been classified as {pred_str}."
            instruction = "Please explain why: "

        return_s = get_few_shot_result(few_shot_str, conversation.custom_input, intro, instruction, conversation, pred_str)
        return return_s, 1

    id_list = []
    for item in parse_text:
        try:
            if type(int(item)) == int:
                id_list.append(int(item))
        except ValueError:
            pass

    dataset = conversation.temp_dataset.contents["X"]
    model = conversation.get_var("model").contents

    if len(conversation.temp_dataset.contents["X"]) == 0:
        return "There are no instances that meet this description!", 0
    results = get_results(dataset_name, data_path)

    # Few-shot setting
    few_shot = True

    return_s = ""
    for idx in id_list:

        instance = dataset.loc[[idx]].values.tolist()[0]

        model_predictions = model.predict(dataset, idx)
        pred = model_predictions[0]
        few_shot_str = ""

        if dataset_name == "boolq":
            text = "Question: " + instance[0] + "\nPassage: " + instance[1]
            label_dict = {0: "false", 1: "true"}
            pred_str = label_dict[pred]
            intro = f"Answer: {pred_str}"
            instruction = "Please explain the answer: "
        elif dataset_name == "daily_dialog":
            text = "Dialog: '" + instance[0] + "'"
            label_dict = conversation.class_names
            pred_str = pred
            other_class_names = ", ".join(
                [label_dict[c] for c in conversation.class_names if label_dict[c] not in [pred, "dummy"]])
            intro = f"The dialogue act of this text has been classified as {pred_str} (over {other_class_names})."
            instruction = "Please explain why: "
        elif dataset_name == "olid":
            text = "Tweet: '" + instance[0] + "'"
            label_dict = {0: "non-offensive", 1: "offensive"}
            pred_str = label_dict[pred]
            intro = f"The tweet has been classified as {pred_str}."
            instruction = "Please explain why: "
        else:
            return f"Dataset {dataset_name} currently not supported by rationalize operation", 1

        if few_shot:
            few_shot_str += get_few_shot_str(gpt_rationales)
        if idx in results['Id']:
            explanation = results.loc[idx]['Explanation']

            if simulation:
                return_s += formalize_output(dataset_name, text)
                return_s += "<br><br><b>Explanation:</b> " + explanation
            else:
                return_s += formalize_output(dataset_name, text) \
                            + "<br><br><b>Prediction:</b> " + pred_str \
                            + "<br><br><b>Explanation:</b> " + explanation
        else:
            return_s += get_few_shot_result(few_shot_str, text, intro, instruction, conversation, pred_str)

        return return_s, 1
