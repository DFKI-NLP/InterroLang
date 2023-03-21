import json
import pandas as pd
import random
import re
import sys
import ast
import string

from num2words import num2words
from nltk.corpus import stopwords

import spacy
nlp = spacy.load("en_core_web_sm")

# select based on the SpaCy chunking
def select_chunks_with_spacy(texts, span_max_len=3, num_samples=20):
    tokens_to_include = []
    for txt in texts:
        doc = nlp(txt)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split())<=3 and len(chunk.text)>1:
                tokens_to_include.append(chunk.text)    
    random.shuffle(tokens_to_include)
    tokens_to_include = list(set(tokens_to_include))
    return tokens_to_include[:num_samples]

# select ngrams randomly
def select_tokens_to_include(texts, span_max_len=3, num_samples=20):
    tokens_to_include = []
    concat_texts = ""
    for txt in texts:
        concat_texts+=txt+" "
    tokens = concat_texts.strip().split()
    num_samples_per_ngram = round(num_samples/span_max_len)
    for i in range(1,span_max_len+1):
        # extract ngrams
        ngrams = [' '.join(tokens[j:j+i]) for j in range(len(tokens)-i)]
        # select random samples
        selected_spans = random.choices(ngrams, k=num_samples_per_ngram)
        selected_spans = [s for s in selected_spans if not(s in string.punctuation) and not("\t"in s) and not(s in stopwords.words())] 
        #selected_spans = [re.escape(s) for s in selected_spans]
        tokens_to_include.extend(selected_spans)
    # remove punctuation in 90% of the cases
    most_cases = round(len(tokens_to_include)*0.9)
    tokens_to_include = [''.join([ch for ch in t if not(ch in string.punctuation)]) for t in tokens_to_include[:most_cases]]+tokens_to_include[most_cases:]
    return tokens_to_include[:num_samples]

def select_ids(id_range=1000, num_samples=20):
    ids = random.choices([i for i in range(1,id_range+1)], k=num_samples)
    most_cases = round(len(ids)*0.9)
    ids = ids[:most_cases]+[num2words(idx) for idx in ids[most_cases:]]
    random.shuffle(ids)
    return ids

def select_numbers(num_range=10, num_samples=20):
    numbers = random.choices([i for i in range(1,num_range+1)], k=num_samples)
    most_cases = round(len(numbers)*0.8)
    numbers = numbers[:most_cases]+[num2words(num) for num in numbers[most_cases:]]
    random.shuffle(numbers)
    return numbers

dataset_file = "dailydialog_dataset/dataset_da_val.csv"
colname = "dialog"
dataset = pd.read_csv(dataset_file)

num_samples = 100
span_max_len = 3
num_range = 10
id_range = 1000 #len(dataset)
# extract the text data
orig_texts = dataset[colname]
if colname=="dialog":
    texts = [" ".join(ast.literal_eval(txt)) for txt in orig_texts]
else:
    texts = orig_texts
include_tokens = select_chunks_with_spacy(texts, span_max_len=span_max_len, num_samples=num_samples)
#include_tokens = select_tokens_to_include(texts, span_max_len=span_max_len, num_samples=num_samples)
ids = select_ids(id_range, num_samples)
numbers = select_numbers(num_range, num_samples)

print('include_tokens:', include_tokens)
print('ids:', ids)
print('numbers:', numbers)

data = {"includetoken":include_tokens,"id":ids,"number":numbers}
json_string = json.dumps(data, indent=2)
with open("templates/slot_values.json", "w") as f:
    print(json_string, file=f)
    #json.dump(json_string, f)
