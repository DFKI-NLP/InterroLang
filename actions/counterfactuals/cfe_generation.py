import torch
from torch.utils.data import DataLoader
from da_model_utils import DANetwork, DADataset, model_id2label
from transformers import BertTokenizer
import sys
# https://huggingface.co/uw-hai/polyjuice
from polyjuice import Polyjuice
from polyjuice.generations.special_tokens import *

# ALL_CTRL_CODES = set([
#    LEXCICAL, RESEMANTIC, NEGATION, INSERT,
#    DELETE, QUANTIFIER, RESTRUCTURE, SHUFFLE
# ])

if torch.cuda.is_available():
    is_cuda = True
    device = 'cuda'
else:
    is_cuda = False
    device = 'cpu'
pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=is_cuda)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def encode_sample(sample):
    encoded = tokenizer.encode_plus(sample, return_tensors='pt').to(device)
    return encoded


def get_samples_from_pj(instance, ctrl_code):
    try:
        generated_samples = pj.perturb(instance, ctrl_code=ctrl_code, num_perturbations=None,
                                       perplex_thred=10)  # , num_beams=5)
    except:
        generated_samples = pj.perturb(instance, ctrl_code=ALL_CTRL_CODES, num_perturbations=None, perplex_thred=None)
    return generated_samples


# model: torch model (e.g., model for DA classification)
# instance: input string
# number: max number of samples to generate
# ctrl_code: ['resemantic', 'restructure', 'negation', 'insert', 'lexical', 'shuffle', 'quantifier', 'delete'] by default uses all codes
# returns two lists with the generated samples that result in the same or different label
# each list consists of tuples (generated_text, class)
# e.g.:
# same label: [('i also have blow if you prefer to do a few shots.', 'directive'), ('i also have blow if you prefer to do this.', 'directive')]
# diff label: [('also blew me away with his single second.', 'inform')]

def cfe(model, instance, number, ctrl_code=ALL_CTRL_CODES, id2label=None):
    new_samples = get_samples_from_pj(instance, ctrl_code)
    encoded_instance = encode_sample(instance)
    orig_prediction = model(encoded_instance['input_ids'], encoded_instance['attention_mask'])
    orig_prediction = torch.argmax(orig_prediction).item()
    if id2label is not None:
        orig_prediction = id2label[orig_prediction]
    same_label_samples = []
    diff_label_samples = []
    for new_sample in new_samples:
        encoded_new_sample = encode_sample(new_sample)
        prediction = model(encoded_new_sample['input_ids'], encoded_new_sample['attention_mask'])
        prediction = torch.argmax(prediction).item()
        if id2label is not None:
            prediction = id2label[prediction]
        if prediction != orig_prediction:
            diff_label_samples.append((new_sample, prediction))
        else:
            same_label_samples.append((new_sample, prediction))
    return same_label_samples[:number], diff_label_samples[:number]


### loading DA classification model for testing ###
model = DANetwork()
model.load_state_dict(torch.load('../da_classifier/saved_model/5e_5e-06lr'))
model = model.to(device)

### testing with some random text ###
instance_text = 'Why is this important?'
same, diff = cfe(model, instance_text, number=4, ctrl_code='lexical')
print('original text:', instance_text)
print('same label predictions:', same)
print('different label predictions:', diff)

### testing with the DA test dataloader ###
test_dataloader = torch.load('../da_classifier/test_dataloader.pth')
for b_input in test_dataloader:
    input_ids = b_input[0].to(device)
    instance_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True))
    same, diff = cfe(model, instance_text, number=4, id2label=model_id2label)
    print('original text:', instance_text)
    print('same label predictions:', same)
    print('different label predictions:', diff)
    input_mask = b_input[1].to(device)
    labels = b_input[2].to(device)
    with torch.no_grad():
        result = model(input_ids, input_mask)
        predicted_label_id = torch.argmax(result.detach().cpu()).item()  # result.logits...
        true_label_id = labels.to('cpu').numpy()[0]
        print('predicted label:', model_id2label[predicted_label_id])
        print('true label:', model_id2label[true_label_id])
    print()

