import torch
from transformers import BertTokenizer

from actions.counterfactuals.cfe_generation_refactor import CFEExplainer, ALL_CTRL_CODES
from explained_models.da_classifier.da_model_utils import DADataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)


def get_dataloader(data, batch_size, dtype):
    samples = []
    for i in range(len(data)):
        d_texts = data[i]['dialog']
        d_labels = data[i]['act']
        assert(len(d_texts)==len(d_labels))
        for j in range(len(d_texts)):
            if j==0:
                prev_text = 'start'
            else:
                prev_text = d_texts[j-1]
            samples.append((prev_text+' [SEP] '+d_texts[j], d_labels[j]))
    dataset = DADataset(samples)
    if dtype=='train':# or dtype=='val':
        dataloader = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = batch_size, num_workers = 4)
    else:
        dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 1)
    return dataloader


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


def get_text_by_id_from_csv(_id):
    import pandas as pd

    df = pd.read_csv('./data/daily_dialog_test.csv')
    text = df["dialog"][_id]
    label = df["act"][_id]
    return text, label

def counterfactuals_operation(conversation, parse_text, i, **kwargs):
    # Parsed: filter id 54 and nlpcfe [E]

    import nltk
    nltk.download('omw-1.4')

    import spacy
    # spacy.load("en_core_web_sm")
    spacy.load('en_core_web_sm')

    _id, cfe_num = extract_id_cfe_number(parse_text)

    from datasets import load_dataset
    test_data = load_dataset('daily_dialog', split='test')
    test_dataloader = get_dataloader(test_data, 1, "test")

    # instance = get_text_by_id(conversation, _id)
    instance, label = get_text_by_id_from_csv(_id)
    # return instance, 1
    # test_dataloader = torch.load('./explained_models/da_classifier/test_dataloader.pth')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # for b_input in test_dataloader:
    #     input_ids = b_input[0].to(device)
    #     instance = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True))
    #     break

    cfe_explainer = CFEExplainer()
    model_id2label = {0: 'dummy', 1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}
    same, diff = cfe_explainer.cfe(instance, cfe_num, ctrl_code=ALL_CTRL_CODES, id2label=model_id2label)

    if len(same) > 0:
        predicted_label = same[0][1]
    else:
        model = conversation.get_var("model").contents
        predicted_label = model(instance)

    return_s = ""
    if len(diff) > 0:
        # [('oh , god , no thanks .', 'dummy'), ('oh , good boy , no thanks .', 'dummy')]
        return_s += f"If the original text: <b>{instance}</b>. <br><br>"
        return_s += "is changed to <br>"
        flipped_label = diff[0][1]

        return_s += "<ul>"
        for i in range(len(diff)):
            return_s += '<li>'
            return_s += diff[i][0]
            return_s += '</li>'
        return_s += "</ul><br>"

        return_s += f"the predicted label <b>{predicted_label}</b> changes to <b>{flipped_label}</b>. <br>"

    else:
        return_s += f"This sentence is always classified as <b>{predicted_label}</b>!"

    return return_s, 1
