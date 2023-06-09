import pandas as pd

from timeout import timeout
import json
import pandas as pd

def get_results(dataset_name,data_path):
    """
    Get the rationlize result

    Args:
        data_path: path to json file
    Returns:
        results: results in csv format
    """
    path =data_path +dataset_name+"/dolly-rationales.csv"
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


@timeout(60)
def rationalize_operation(conversation, parse_text, i, simulation, data_path="./cache/", **kwargs):
    if not conversation.decoder.gpt_parser_initialized:
        return f"Rationalize operation not enabled for {conversation.decoder.parser_name}"

    # TODO: Custom input – if conversation.used and conversation.custom_input:

    id_list = []
    for item in parse_text:
        try:
            if type(int(item)) == int:
                id_list.append(int(item))
        except ValueError:
            pass

    dataset_name = conversation.describe.get_dataset_name()
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
            gpt_rationales = "cache/boolq/GPT-3.5_rationales_BoolQ_val_400.csv"

        elif dataset_name == "daily_dialog":
            text = "Text: '" + instance[0] + "'"
            label_dict = conversation.class_names
            pred_str = pred
            other_class_names = ", ".join(
                [label_dict[c] for c in conversation.class_names if label_dict[c] not in [pred, "dummy"]])
            intro = f"The dialogue act of this text has been classified as {pred_str} (over {other_class_names})."
            instruction = "Please explain why: "
            gpt_rationales = "cache/daily_dialog/GPT-4_rationales_DD_test_200.csv"

        elif dataset_name == "olid":
            text = "Tweet: '" + instance[0] + "'"
            label_dict = {0: "non-offensive", 1: "offensive"}
            pred_str = label_dict[pred]
            intro = f"The tweet has been classified as {pred_str}."
            instruction = "Please explain why: "
            gpt_rationales = "cache/olid/GPT-4_rationales_OLID_val_132.csv"

        else:
            return f"Dataset {dataset_name} currently not supported by rationalize operation", 1

        if few_shot:
            few_shot_str += get_few_shot_str(gpt_rationales)
        if idx in results['Id']:
            inputs = text
            explanation = results.loc[idx]['Explanation']

            if simulation:
                return_s += "<b>Original text:</b> " + text \
                            + "<br><b>Explanation:</b> " + explanation
            else:
                return_s += "<b>Original text:</b> " + text \
                            + "<br><b>Prediction:</b> " + pred_str \
                            + "<br><b>Explanation:</b> " + explanation
        else:
            prompt = f"{few_shot_str}" \
                     f"{text}\n" \
                     f"{intro}\n" \
                     f"{instruction}\n"
            print(f"[rationalize operation]\n=== PROMPT ===\n{prompt}")

            input_ids = conversation.decoder.gpt_tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(device = "cpu")
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

        return return_s, 1
