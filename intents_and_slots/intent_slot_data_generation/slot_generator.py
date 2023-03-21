import random
import re
import json
import sys
import pandas as pd

import nlpaug.augmenter.word as naw
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

non_token_chars = [" "]

def update_token_annotation(token_annotation):
    if "B" in token_annotation:
        token_annotation = "B"
    elif "I" in token_annotation:
        token_annotation = "I"
    else:
        token_annotation = "O"
    return token_annotation

def convert_to_token_level(txt, annotations):
    token_annotations = dict()
    for ak, annotation in annotations.items():
        token_annotations[ak] = []
        tokens = []
        cur_token = []
        cur_annotation = []
        assert(len(txt)==len(annotation))
        for ci, ch in enumerate(txt):
            if ch in non_token_chars:
                if len(cur_token)>0:
                    tokens.append("".join(cur_token))
                    token_annotations[ak].append(update_token_annotation("".join(cur_annotation)))
                    cur_token = []
                    cur_annotation = []
            else:
                cur_token.append(ch)
                cur_annotation.append(annotation[ci])
        if len(cur_token)>0:
            tokens.append("".join(cur_token))
            token_annotations[ak].append(update_token_annotation("".join(cur_annotation)))
    return tokens, token_annotations


# Generator for intents and slots based on the templates

def fill_template(template_file, slots2values, fill_rounds=2, augmentation_rounds=0):
    templates = []
    filled_templates = []
    with open(template_file) as f:
        templates = f.readlines()  
    for template in templates:
        for fround in range(fill_rounds):
            txt = template.strip()
            slot_to_fill = re.search("\{[a-z]+\}", txt)
            annotations = dict()
            while slot_to_fill:
                annotation = ["O"]*len(txt)
                slot = txt[slot_to_fill.start()+1:slot_to_fill.end()-1]
                if not(slot in slots2values):
                    raise Exception("Invalid slot type:", slot)
                else:
                    # get a random slot value and add char-wise annotation
                    # TA: shall we use weights to get more likely slot values more often?
                    slot_value = random.choice(slots2values[slot])
                    mstart = slot_to_fill.start()
                    mend = slot_to_fill.end()
                    slot_anno = []
                    for sidx, stoken in enumerate(slot_value):
                        if sidx==0:
                            slot_anno.append("B")
                        else:
                            slot_anno.append("I")
                    if mend<len(annotation):
                        annotation = annotation[:mstart]+slot_anno+annotation[mend:]
                    else:
                        annotation = annotation[:mstart]+slot_anno
                    annotations[slot] = annotation
                    txt = txt[:mstart] + str(slot_value) + txt[mend:]
                    slot_to_fill = re.search("\{[a-z]+\}", txt)
            tokens, annotations = convert_to_token_level(txt, annotations)
            filled_templates.append((tokens, annotations))
            if augmentation_rounds>0:
                # TA: augmentation is not very reliable
                # perhaps checking the similarity with SentenceTransformers would help?
                for aug_round in range(augmentation_rounds):
                    new_txt = aug.augment(txt)
                    new_tokens = new_txt.split()
                    # make sure we have the same amount of tokens
                    if len(new_tokens)==len(tokens):
                        filled_templates.append((new_tokens, annotations))

    return filled_templates


f = open("templates/slot_values.json")
slots2values = json.load(f)
f.close()

dtypes = ["train", "dev", "test"]
template_dir = "templates/"
template_files = ["include.txt"]#, "nlpcfe.txt", "predict.txt"]
generated_samples = dict()
for template_file in template_files:
    filled_templates = fill_template(template_dir+template_file, slots2values, fill_rounds=5, augmentation_rounds=0)
    for tmp in filled_templates:
        txt, annotations = tmp
        # store the generated samples for each slot/annotation type in a separate file
        for anno_type, anno in annotations.items():
            if not(anno_type in generated_samples):
                generated_samples[anno_type] = {"text":[], "labels":[]}
            generated_samples[anno_type]["text"].append(" ".join(txt))
            generated_samples[anno_type]["labels"].append(" ".join(anno))            
            #for i in range(len(txt)):
            #    if "Generally" in txt:
            #        print(txt[i], anno[i])
for anno_type in generated_samples.keys():
    data_splits = {"train":dict(), "dev":dict(), "test":dict()}
    total_samples = len(generated_samples[anno_type]["text"])
    orig_anno_texts = generated_samples[anno_type]["text"]
    orig_anno_labels = generated_samples[anno_type]["labels"]
    # shuffle the data
    indices = [i for i in range(len(orig_anno_texts))]
    random.shuffle(indices)
    anno_texts = []
    anno_labels = []
    for i in range(len(indices)):
        anno_texts.append(orig_anno_texts[indices[i]]) 
        anno_labels.append(orig_anno_labels[indices[i]])
    train_samples = round(0.75*total_samples)
    dev_samples = train_samples+round((total_samples-train_samples)/2)
    data_splits["train"]["text"] = anno_texts[:train_samples]
    data_splits["train"]["labels"] = anno_labels[:train_samples]
    data_splits["dev"]["text"] = anno_texts[train_samples:dev_samples]
    data_splits["dev"]["labels"] = anno_labels[train_samples:dev_samples]
    data_splits["test"]["text"] = anno_texts[dev_samples:]
    data_splits["test"]["labels"] = anno_labels[dev_samples:]

    for dtype in dtypes:
        df = pd.DataFrame.from_dict(data_splits[dtype])
        df.to_csv("../csv_slots/"+anno_type+"_"+dtype+".csv", sep=',', encoding='utf-8')
             
             
