import numpy as np
import openai
import pandas as pd
import yaml
from pprint import pprint
from tqdm import tqdm


def rationalize_operation(conversation, parse_text, i, **kwargs):
    id_list = []
    for item in parse_text:
        try:
            if int(item):
                id_list.append(int(item))
        except ValueError:
            pass

    dataset_name = conversation.describe.get_dataset_name()
    dataset = conversation.temp_dataset.contents['X']
    model = conversation.get_var('model').contents

    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    return_s = ''
    for idx in id_list:
        instance = dataset.loc[[idx]].values.tolist()[0]

        if dataset_name == "boolq":
            text = 'Question: ' + instance[0] + '\nPassage: ' + instance[1]
            label_dict = {0: 'false', 1: 'true'}
            text_description = 'question and passage'
            fields = ['question', 'passage']
            fields_enum = ', '.join([f"'{f}'" for f in fields])
            output_description = 'answer'
            model_predictions = model.predict(dataset, idx)
            pred_str = label_dict[model_predictions[0]]
        else:
            return f"Dataset {dataset_name} currently not supported by rationalize operation", 1

        prompt = f"{text}\n" \
                 f"Based on {text_description}, the {output_description} is {pred_str}. " \
                 f"Without using {fields_enum}, or revealing the answer or outcome in your response, " \
                 f"explain why: "

        input_ids = conversation.decoder.gpt_tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device = "cpu")
        generation = conversation.decoder.gpt_model.generate(
            input_ids,
            max_length=350,
            no_repeat_ngram_size=2,
        )
        decoded_generation = conversation.decoder.gpt_tokenizer.decode(generation[0], skip_special_tokens=True)

        inputs = decoded_generation.split("Based on ")[0]
        explanation = decoded_generation.split("explain why: ")[1]

        return_s += inputs + "<br><b>Explanation:</b> " + explanation

    return return_s, 1


def generate_rationales_from_api(
        openai_model_id: str,
        data: pd.DataFrame,
        label_dict: dict,
        fields: list[str],
        text_description: str,
        output_description: str,
    ):
    handler = ChatGPTHandler(model=openai_model_id)
    handler.use_wandb()
    for i, instance in tqdm(data.iterrows(), total=len(data)):
        try:
            idx = instance['idx']
            text = instance['text']
            prediction = instance['prediction']
        except KeyError:
            raise "Dataset does not contain the required idx, text, and prediction columns."

        pred_str = label_dict[prediction]

        fields_enum = ', '.join([f"'{f}'" for f in fields])

        prompt = f"{text}\n" \
                 f"Based on {text_description}, the {output_description} is {pred_str}. " \
                 f"Without using {fields_enum}, or revealing the answer or outcome in your response, " \
                 f"explain why: "
        handler.chat_request(prompt, id=idx)

    handler.visualize()


class ChatGPTHandler:
    def __init__(self, model: str = None):
        with open("../../configs/openai_api_key.yaml", "r") as stream:
            openai_credentials = yaml.safe_load(stream)
        openai.api_key = openai_credentials["api_key"]
        openai.organization = openai_credentials["organization"]

        self.avail_models = []
        self.responses = []
        self.wandb_true = False
        self.model = model

    def use_wandb(self) -> None:
        import wandb
        self.wandb_true = True
        wandb.init(project="InterroLang_Rationales")
        self.prediction_table = wandb.Table(columns=["prompt", "id", "completion"])

    def list_models(self) -> list:
        models = openai.Model.list()
        for i in models["data"]:
            self.avail_models.append(i["id"])
        print(self.avail_models)
        return self.avail_models

    def chat_request(self, prompt: str = None, temperature: float = 0.5, max_tokens: int = 300, top_p: float = 1.0,
                     stop: str = "\n", id=None) -> list:
        if prompt is not None:
            prompt = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        self.responses.append(response)

        if self.wandb_true:
            self.prediction_table.add_data(prompt, id, response["choices"][0]["message"]["content"])
        return response

    def visualize(self) -> None:
        if self.wandb_true:
            import wandb
            wandb.log({"predictions": self.prediction_table})
            wandb.finish()
        else:
            pprint(self.responses)


if __name__ == "__main__":
    data = pd.read_csv("../../data/boolq_validation.csv")
    explanations = pd.read_json("../../cache/boolq/ig_explainer_boolq_explanation.json")

    data['text'] = 'Question: ' + data['question'] + '\nPassage: ' + data['passage']

    label_mapping = {0: 'false',
                     1: 'true'}

    text_desc = 'question and passage'
    text_fields = ['question', 'passage']
    out_desc = 'answer'

    data['prediction'] = [np.argmax(x) for x in explanations['predictions']]

    # Reduce amount of data
    data = data[:400]

    model_id = "gpt-3.5-turbo"
    generate_rationales_from_api(model_id, data, label_mapping, text_fields, text_desc, out_desc)
