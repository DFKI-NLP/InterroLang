import os
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AutoTokenizer

from actions.explanation.feature_importance import FeatureAttributionExplainer
from explained_models.DataLoaderABC.hf_dataloader import HFDataloader
from explained_models.ModelABC.distilbert_qa_boolq import DistilbertQABoolModel
from explained_models.Tokenizer.tokenizer import HFTokenizer


def initiate_ig_explainer(data_path, if_generate=False):
    """
    Initialization of explainer

    Args:
        if_generate (bool, optional): if needed to generate IG. Defaults to False.
        data_path: path to json file
    Returns:
        Explainer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "andi611/distilbert-base-uncased-qa-boolq"
    tokenizer = HFTokenizer(model_id)
    dataloader = HFDataloader(tokenizer=tokenizer.tokenizer, batch_size=1, number_of_instance=10)
    model = DistilbertQABoolModel(dataloader, num_labels=2, model_id=model_id)
    explainer = FeatureAttributionExplainer(model, device=device)
    if if_generate:
        explainer.generate_explanation(store_data=if_generate, data_path=data_path)

    return explainer


def get_results(explainer, data_path):
    """
    Get the IG result

    Args:
        explainer (string): string of explainer name
        data_path: path to json file
    Returns:
        results: results in JSON format
        model: explainer model
    """
    if explainer == "ig_explainer":
        # Check if IG explanation is already generated
        model = initiate_ig_explainer(data_path, if_generate=(not os.path.exists(data_path)))

        fileObject = open(data_path, "r")
        jsonContent = fileObject.read()
        results = json.loads(jsonContent)
    return results, model


def results_with_pattern(results):
    """
    Output the results with certain pattern

    Args:
        results: attribution scores list
    Returns:
        None
    """
    # example: dumb, fucking, and ugly are the most attributed for the hate speech label
    if len(results) == 1:
        return results[0][0] + " is the most attributed"
    else:
        string = ""
        for i in range(len(results) - 1):
            string += results[i][0] + ", "
        string += "and "
        string += results[len(results) - 1][0]
        return string + " are the most attributed."


def topk(conversation, explainer, k, threshold=-1, data_path="../../cache/boolq/ig_explainer_boolq_explanation.json",
         res_path="../../cache/boolq/ig_explainer_boolq_attribution.json", print_with_pattern=True, class_idx=None):
    """
    The operation to get most k important tokens

    Args:
        conversation: conversation object
        explainer (string): string of explainer name
        k (int): number of tokens
        threshold (int, optional): Threshold of #occurance of a single token. Defaults to -1.
        data_path: path to json file
        res_path: path to store attribution scores
        print_with_pattern: if output the results using the certain pattern
        class_idx: filter label (index)
    Returns:
        sorted_scores: top k important tokens
    """
    if os.path.exists(res_path) and threshold == -1 and (class_idx is None):
        fileObject = open(res_path, "r")
        jsonContent = fileObject.read()
        result_list = json.loads(jsonContent)

        if len(result_list) >= k:
            if print_with_pattern:
                return results_with_pattern(result_list[:k])
            else:
                return result_list[:k]
        else:
            print("[Info] The length of score is smaller than k")
            if print_with_pattern:
                return results_with_pattern(result_list)
            else:
                return result_list

    if "boolq" in data_path:
        results, model = get_results(explainer=explainer, data_path=data_path)
        tokenizer = model.model.tokenizer
    elif "daily_dialog" in data_path:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        fileObject = open(data_path, "r")
        jsonContent = fileObject.read()
        results = json.loads(jsonContent)
    elif "olid" in data_path:
        tokenizer = AutoTokenizer.from_pretrained("sinhala-nlp/mbert-olid-en")
        fileObject = open(data_path, "r")
        jsonContent = fileObject.read()
        results = json.loads(jsonContent)
    else:
        pass

    # individual tokens
    word_set = set()
    word_counter = {}
    word_attributions = {}

    if class_idx:
        #print('class name: ', class_idx)
        temp = []
        for res in results:
            if "daily_dialog" in data_path:
                if res["label"] == conversation.class_names[class_idx]:
                    temp.append(res)
            else:
                if res["label"] == class_idx:
                    temp.append(res)
    else:
        temp = results

    pbar = tqdm(temp)

    for result in pbar:
        pbar.set_description('Processing Attribution')
        attribution = result["attributions"]
        tokens = list(tokenizer.convert_ids_to_tokens(result["input_ids"]))
        counter = 0

        # count for attributions and #occurance
        for token in tokens:
            if not token in word_set:
                word_set.add(token)
                word_counter[token] = 1
                word_attributions[token] = attribution[counter]
            else:
                word_counter[token] += 1
                word_attributions[token] += attribution[counter]
            counter += 1

    scores = {}
    if threshold == -1:
        for word in word_set:
            scores[word] = (word_attributions[word] / word_counter[word])
    else:
        for word in word_set:
            if word_counter[word] >= threshold:
                scores[word] = (word_attributions[word] / word_counter[word])

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if not os.path.exists(res_path):
        jsonString = json.dumps(sorted_scores)
        jsonFile = open(res_path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    if len(sorted_scores) >= k:
        if print_with_pattern:
            return results_with_pattern(sorted_scores[:k])
        else:
            return sorted_scores[:k]
    else:
        print("[Info] The length of score is smaller than k")
        if print_with_pattern:
            return results_with_pattern(sorted_scores)
        else:
            return sorted_scores
