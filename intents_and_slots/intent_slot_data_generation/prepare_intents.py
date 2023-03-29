import pandas as pd
import random
import sys
import json

intent = "include"
rounds_per_template_pos = 35 #50 for all but include, 35 for include
rounds_per_template_neg = 50

all_intents = ["include", "nlpcfe", "similar", "predict"]
all_slots = {"include":["includetoken"], "nlpcfe":["id", "number"], "similar":["id","number"], "predict":["id"]}

# create dataframes
data = []
with open("templates/"+intent+".txt") as f:
    data = f.readlines()
print(data)

f = open("templates/slot_values.json")
slots = json.load(f)
f.close()
for slot in all_slots[intent]:
    print(slots[slot])


# create the data with random replacements

all_samples = []
for sample in data:
    tround = 0
    while tround < rounds_per_template_pos:
        for slot in all_slots[intent]:
            random_slot_val = random.choice(slots[slot])
            sample = sample.replace("{"+slot+"}", str(random_slot_val))
        all_samples.append((sample.strip(),1))
        tround+=1

# fetch negative samples (other intents)
total_samples = len(all_samples)
max_neg_samples = 2*total_samples#+round(total_samples/2)
neg_samples = []
for other_intent in all_intents:
    if other_intent==intent:
        continue
    other_intent_data = []
    with open("templates/"+other_intent+".txt") as f:
        other_intent_data = f.readlines()
    for sample in other_intent_data:
        #if len(neg_samples)>max_neg_samples:
        #    break
        tround = 0
        while tround < rounds_per_template_neg:
            for slot in all_slots[other_intent]:
                random_slot_val = random.choice(slots[slot])
                sample = sample.replace("{"+slot+"}", str(random_slot_val))
            neg_samples.append((sample.strip(),0))
            tround+=1
            
random.shuffle(neg_samples)
all_samples.extend(neg_samples[:max_neg_samples])

# prepare the data
random.shuffle(all_samples)
print(all_samples)

test_tokens = []
test_labels = []
dev_tokens = []
dev_labels = []
train_tokens = []
train_labels = []

test_limit = round(0.2*len(all_samples))
dev_limit = test_limit+round(0.1*(len(all_samples)-test_limit))

for i, sample in enumerate(all_samples):
    if i<test_limit:
        test_tokens.append(sample[0])
        test_labels.append(sample[1])
    elif i>=test_limit and i<dev_limit:
        dev_tokens.append(sample[0])
        dev_labels.append(sample[1])
    if i<test_limit:
        train_tokens.append(sample[0])
        train_labels.append(sample[1])

print("test:", test_tokens, len(test_tokens))
print("dev:", dev_tokens, len(dev_tokens))
print("train:", train_tokens, len(train_tokens))

# save the data

test_df = pd.DataFrame.from_dict({"text":test_tokens, "labels":test_labels})

dataset_file = "csv_intents/"+intent+"_test.csv"
test_df.to_csv(dataset_file, index=False)

dev_df = pd.DataFrame.from_dict({"text":dev_tokens, "labels":dev_labels})

dataset_file = "csv_intents/"+intent+"_dev.csv"
dev_df.to_csv(dataset_file, index=False)

train_df = pd.DataFrame.from_dict({"text":train_tokens, "labels":train_labels})

dataset_file = "csv_intents/"+intent+"_train.csv"
train_df.to_csv(dataset_file, index=False)



