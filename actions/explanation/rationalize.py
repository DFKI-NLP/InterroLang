import pandas as pd

def rationalize_operation(conversation, parse_text, i, **kwargs):
    if not conversation.decoder.gpt_parser_initialized:
        return f"Rationalize operation not enabled for {conversation.decoder.parser_name}"

    id_list = []
    for item in parse_text:
        try:
            if int(item):
                id_list.append(int(item))
        except ValueError:
            pass

    dataset_name = conversation.describe.get_dataset_name()
    dataset = conversation.temp_dataset.contents["X"]
    model = conversation.get_var("model").contents

    if len(conversation.temp_dataset.contents["X"]) == 0:
        return "There are no instances that meet this description!", 0

    # Few-shot settings
    few_shot = True
    num_shots = 5

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
            max_length = 150  # The 'passage' is usually very long, so we use a longer max_length for this dataset

            # Few-shot
            if few_shot:
                gpt_rationales = pd.read_csv("cache/boolq/GPT-3.5_rationales_BoolQ_val_400.csv")
                for i, row in gpt_rationales.iterrows():
                    few_shot_str += row["prompt"] + row["completion"] + "\n"
                    if i == num_shots - 1:
                        break

        elif dataset_name == "daily_dialog":
            text = "Text: '" + instance[0] + "'"
            label_dict = conversation.class_names
            pred_str = pred
            other_class_names = ", ".join(
                [label_dict[c] for c in conversation.class_names if label_dict[c] not in [pred, 'dummy']])
            intro = f"The dialogue act of this text has been classified as {pred_str} (over {other_class_names})."
            instruction = "Please explain why: "
            max_length = 150

        elif dataset_name == "olid":
            text = "Tweet: '" + instance[0] + "'"
            label_dict = {0: "non-offensive", 1: "offensive"}
            pred_str = label_dict[pred]
            intro = f"The tweet has been classified as {pred_str}."
            instruction = "Please explain why: "
            max_length = 150

        else:
            return f"Dataset {dataset_name} currently not supported by rationalize operation", 1

        prompt = f"{few_shot_str}" \
                 f"{text}\n" \
                 f"{intro}\n" \
                 f"{instruction}\n"
        print(f"[rationalize operation]\n=== PROMPT ===\n{prompt}")

        input_ids = conversation.decoder.gpt_tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device = "cpu")
        generation = conversation.decoder.gpt_model.generate(
            input_ids,
            max_length=input_ids.size()[-1] + max_length,
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
