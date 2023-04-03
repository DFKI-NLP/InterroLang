import json
import os

import torch
from explained_models.Explainer.explainer import Explainer

from explained_models.Tokenizer.tokenizer import HFTokenizer
from explained_models.ModelABC.DANetwork import DANetwork

# https://huggingface.co/uw-hai/polyjuice
from polyjuice import Polyjuice
from polyjuice.generations.special_tokens import *
from explained_models.da_classifier.da_model_utils import DADataset

from actions.counterfactuals.nlpaug_util import *


ALL_CTRL_CODES = set([
   LEXCICAL, RESEMANTIC, NEGATION, INSERT,
   DELETE, QUANTIFIER, RESTRUCTURE, SHUFFLE
])


class CFEExplainer(): #Explainer
    def __init__(self, explainer="polyjuice", tokenizer="bert"):
        super(CFEExplainer, self).__init__()
        self.device = None
        self.is_cuda = None
        self.check_cuda()
        
        self.explainer_name = explainer
        if explainer=="polyjuice":
            self.explainer = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=self.is_cuda)
        elif explainer=="nlpaug":
            self.explainer = CustomContextualWordEmbsAug(action="substitute", device=self.device, model_path="bert-base-cased", aug_max=3)
            #self.explainer = naw.ContextualWordEmbsAug(model_path="bert-base-cased", action="substitute")
        else:
            self.explainer = explainer
        self.tokenizer = HFTokenizer('bert-base-uncased', mode='bert').tokenizer if tokenizer == 'bert' else tokenizer

    #def encode_sample(self, sample):
    #    encoded = self.tokenizer.encode_plus(sample, return_tensors='pt').to(self.device)
    #    return encoded

    def get_samples_from_pj(self, instance, ctrl_code):
        try:
            generated_samples = self.explainer.perturb(instance, ctrl_code=ctrl_code, num_perturbations=None, perplex_thred=10)  # , num_beams=5)
        except:
            generated_samples = self.explainer.perturb(instance, ctrl_code=ALL_CTRL_CODES, num_perturbations=None, perplex_thred=None)
        if not(isinstance(generated_samples, list)):
            generated_samples = [generated_samples]
        orig_tokens = []
        aug_tokens = []
        orig_tokens_sample = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instance))).split()
        for si, sample in enumerate(generated_samples):
            aug_tokens_sample = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sample))).split()
            orig_tokens.append(orig_tokens_sample)
            aug_tokens.append(aug_tokens_sample)
        assert(len(generated_samples)==len(aug_tokens))
        return [generated_samples, aug_tokens, orig_tokens]

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
    # ctrl_code: ['resemantic', 'restructure', 'negation', 'insert', 'lexical', 'shuffle', 'quantifier', 'delete'] by default uses all codes
    # returns two lists with the generated samples that result in the same or different label
    # each list consists of tuples (generated_text, class)
    # e.g.:
    # same label: [('i also have blow if you prefer to do a few shots.', 'directive'), ('i also have blow if you prefer to do this.', 'directive')]
    # diff label: [('also blew me away with his single second.', 'inform')]
    def cfe(self, instance, model, number, ctrl_code=['lexical', 'shuffle', 'insert', 'delete'], id2label=None):
        if self.explainer_name=="polyjuice":
             new_samples, aug_tokens, orig_tokens = self.get_samples_from_pj(instance, ctrl_code=ctrl_code)
        elif self.explainer_name=="nlpaug":
            instances = (number*2)*[instance]# 2x cfes for similar/diff predictions
            new_samples, aug_tokens, orig_tokens = self.get_samples_from_nlpaug(instances)
        else:
            raise Exception("Unknown explainer:", self.explainer_name)
        orig_prediction = model.predict([instance])
        same_label_samples = []
        diff_label_samples = []
        for si, new_sample in enumerate(new_samples):
            if new_sample==instance:
                continue
            prediction = model.predict([new_sample])
            out_str = self.get_changed_tokens(orig_tokens[si], aug_tokens[si])
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
                    if orig_tokens[oi]==aug_tokens[ai]:
                        out_str+=aug_tokens[ai]+" "
                        ai+=1
                        oi+=1
                        break
                    out_str+="<b>"+aug_tokens[ai]+"</b> "
                    ai+=1
        return out_str



def nlpcfe_operation(conversation, parse_text, i, max_num_preds_to_print=1, **kwargs):
    return_s = ""
    number = 1
    id_val = -1
    if len(parse_text)>2:
        if len(parse_text)>3: # number and id
            try:                
                if len(parse_text[i+1])!=0:
                    number = int(parse_text[i+1])
                if len(parse_text[i+2])!=0:
                    id_val = int(parse_text[i+2])
                else:
                    return "Could you specify the id, please?", 1

            except:
                return "Could you repeat the id and nuber of counterfactuals please?", 1
        else: # only id
            try:
                id_val = int(parse_text[i+1])
            except:
                return "Could you repeat please? I didn't get the id.", 1
        model = conversation.get_var('model').contents
        id2label = conversation.get_var('dataset').contents["id2label"]
        instance_text = conversation.get_var('dataset').contents["X"].iloc[id_val]["text"]
        same, diff = CFEExplainer("nlpaug").cfe(instance=instance_text, model=model, number=number, id2label=id2label)
        return_s = "The original sample was: " + instance_text + "<br>"
        if len(same)>0:
            return_s += "I found the following counterfactuals that results in the same prediction:<br>"
            for same_pred in same:
                return_s+=str(same_pred[1])+": "
                out_str = same_pred[2]
                if len(out_str)>0:
                    return_s+=out_str+"<br>"
        else:
            return_s += "I couldn't generate any samples that result in the same prediction"
        if len(diff)>0:
            return_s += "I found the following counterfactuals that results in a different prediction:<br>"        
            for diff_pred in diff:
                return_s+=str(diff_pred[1])+": "
                out_str = diff_pred[2]
                if len(out_str)>0:
                    return_s+=out_str+"<br>"
        else:
            return_s += "I couldn't generate any sampels that result in a different prediction"
    else:
        return_s = "Sorry, I don't know for which instance I should provide counterfactuals, please try again."
    
    return return_s, 1



#if __name__ == "__main__":
#    CFEExplainer().generate_explanation()


