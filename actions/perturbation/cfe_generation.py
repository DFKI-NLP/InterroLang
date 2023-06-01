import json
import numpy as np
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from explained_models.Explainer.explainer import Explainer
from explained_models.ModelABC.DANetwork import DANetwork
from explained_models.Tokenizer.tokenizer import HFTokenizer

from actions.perturbation.nlpaug_util import *

class CFEExplainer(Explainer):
    def __init__(self, dataset_name=None):
        super(CFEExplainer, self).__init__()
        self.device = None
        self.is_cuda = None
        self.check_cuda()
        self.explainer =  CustomContextualWordEmbsAug(action="substitute", device=self.device, model_path="bert-base-cased", aug_max=15)
        self.dataset_name = dataset_name

        if dataset_name == 'boolq':
            self.model = AutoModelForSequenceClassification.from_pretrained("andi611/distilbert-base-uncased-qa-boolq", num_labels=2)
            self.tokenizer = HFTokenizer("andi611/distilbert-base-uncased-qa-boolq").tokenizer
        elif dataset_name == 'daily_dialog':
            self.model = DANetwork()
            self.tokenizer = HFTokenizer('bert-base-uncased', mode='bert').tokenizer
        elif dataset_name == 'olid':
            self.model = AutoModelForSequenceClassification.from_pretrained("sinhala-nlp/mbert-olid-en")
            self.tokenizer = AutoTokenizer.from_pretrained("sinhala-nlp/mbert-olid-en")
        else:
            raise NotImplementedError(f"The dataset {self.dataset_name} is not supported!")

        if self.is_cuda:
            self.model.to(self.device)

    def encode_sample(self, sample):
        encoded = self.tokenizer.encode_plus(sample, return_tensors='pt').to(self.device)
        return encoded

    def get_samples_from_nlpaug(self, instances):
        aug_results = self.explainer.augment(instances, n=5)
        generated_samples = []
        aug_tokens = []
        orig_tokens = []
        for aug_res in aug_results:
            generated_samples.append(aug_res[0])
            aug_tokens.append(aug_res[2])
            orig_tokens.append(aug_res[3])
        assert(len(generated_samples)==len(aug_tokens))
        return [generated_samples, aug_tokens, orig_tokens]

    # instance: input string
    # number: max number of samples to generate
    # returns two lists with the generated samples that result in the same or different label
    # each list consists of tuples (generated_text, class)
    # e.g.:
    # same label: [('i also have blow if you prefer to do a few shots.', 'directive'), ('i also have blow if you prefer to do this.', 'directive')]
    # diff label: [('also blew me away with his single second.', 'inform')]
    def cfe(self, instance, number, _id=None):
        instances = (number*2)*[instance]# 2x cfes for similar/diff predictions
        new_samples, aug_tokens, orig_tokens = self.get_samples_from_nlpaug(instances)

        if self.dataset_name == 'boolq':
            if _id is not None:
                import json
                fileObject = open("./cache/boolq/ig_explainer_boolq_explanation.json", "r")
                jsonContent = fileObject.read()
                json_list = json.loads(jsonContent)
                item = json_list[_id]
                orig_prediction = np.argmax(item["predictions"])

            model_id2label = {0: 'False', 1: 'True'}
        elif self.dataset_name == "daily_dialog":
            encoded_instance = self.encode_sample(instance)
            orig_prediction = self.model(encoded_instance['input_ids'], encoded_instance['attention_mask'])
            orig_prediction = torch.argmax(orig_prediction).item()
            model_id2label = {0: 'dummy', 1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}
        elif self.dataset_name == 'olid':
            if _id is not None:
                import json
                fileObject = open("./cache/olid/ig_explainer_olid_explanation.json", "r")
                jsonContent = fileObject.read()
                json_list = json.loads(jsonContent)
                item = json_list[_id]
                orig_prediction = item["predictions"]
            model_id2label = {0: 'False', 1: 'True'}
        else:
            pass

        orig_prediction = model_id2label[orig_prediction]
        same_label_samples = []
        diff_label_samples = []
        for s_i, new_sample in enumerate(new_samples):
            encoded_new_sample = self.encode_sample(new_sample[0])
            aug_tokens_sample = aug_tokens[s_i]
            orig_tokens_sample = orig_tokens[s_i]

            prediction = self.model(encoded_new_sample['input_ids'], encoded_new_sample['attention_mask'])

            if self.dataset_name == 'boolq':
                prediction = np.argmax(prediction.logits[0].cpu().detach().numpy())
            elif self.dataset_name == "daily_dialog":
                prediction = torch.argmax(prediction).item()
            elif self.dataset_name == 'olid':
                prediction = np.argmax(prediction.logits[0].cpu().detach().numpy())
            else:
                pass

            prediction = model_id2label[prediction]
            out_str = self.get_changed_tokens(orig_tokens[s_i], aug_tokens[s_i])
            if prediction != orig_prediction:
                diff_label_samples.append((new_sample, prediction, out_str))
            else:
                same_label_samples.append((new_sample, prediction, out_str))
        return same_label_samples[:number], diff_label_samples[:number]

    def check_cuda(self):
        if torch.cuda.is_available():
            self.is_cuda = True
            self.device = 'cuda'
        else:
            self.is_cuda = False
            self.device = 'cpu'


    def get_changed_tokens(self, orig_tokens, aug_tokens):
        oi = 0
        ai = 0
        out_str = ""
        while ai < len(aug_tokens):
            if oi>=len(orig_tokens) or orig_tokens[oi]==aug_tokens[ai]:
                out_str+=aug_tokens[ai]+" "
                ai+=1
                oi+=1
            else:
                out_str+="<b>"+aug_tokens[ai]+"</b> "
                ai+=1
                while not(ai>=len(aug_tokens) or oi>=len(orig_tokens) or orig_tokens[oi]==aug_tokens[ai]):
                    # heuristics to check that oi token appears in aug_tokens
                    while (oi<len(orig_tokens) and not(orig_tokens[oi] in aug_tokens[ai:ai+3])):
                        oi+=1
                    if oi < len(orig_tokens) and ai < len(orig_tokens) and orig_tokens[oi]==aug_tokens[ai]:
                        out_str+=aug_tokens[ai]+" "
                        ai+=1
                        oi+=1
                        break
                    out_str+="<b>"+aug_tokens[ai]+"</b> "
                    ai+=1
        return out_str



    def generate_explanation(self, store_data=False,
                             data_path="../../cache/daily_dialog/cfe_daily_dialog_explanation.json"):
        if os.path.exists(data_path):
            fileObject = open(data_path, "r")
            jsonContent = fileObject.read()
            result_list = json.loads(jsonContent)
        else:
            json_list = []
            ### testing with the DA test dataloader ###
            test_dataloader = torch.load('../../explained_models/da_classifier/test_dataloader.pth')
            for b_input in test_dataloader:
                input_ids = b_input[0].to(self.device)
                instance_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True))
                same, diff = self.cfe(instance_text, number=4, id2label=model_id2label)
                input_mask = b_input[1].to(self.device)
                labels = b_input[2].to(self.device)

                with torch.no_grad():
                    result = self.model(input_ids, input_mask)
                    predicted_label_id = torch.argmax(result.detach().cpu()).item() # result.logits...
                    true_label_id = labels.to('cpu').numpy()[0]

                json_list.append({
                    "instance_text": instance_text,
                    "same_label_predictions": same,
                    "different_label_predictions": diff,
                    "predicted_label": model_id2label[predicted_label_id],
                    "true_label": model_id2label[true_label_id]
                })

            jsonString = json.dumps(json_list)
            jsonFile = open(data_path, "w")
            jsonFile.write(jsonString)
            jsonFile.close()


if __name__ == "__main__":
    CFEExplainer().generate_explanation()
