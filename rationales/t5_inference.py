from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_helper import load_raw_dataset, Data_Collator_for_Training, get_tensor_dataset

from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from random import randrange
from transformers import AutoModelForSeq2SeqLM
model_id="google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset("csv", data_files="../cache/boolq/GPT-3.5_rationales_BoolQ_val_400.csv")
dataset=dataset["train"].train_test_split()
model = AutoModelForSeq2SeqLM.from_pretrained("/hd2/sahil/t5/checkpoint-95")

sample = dataset['test'][randrange(len(dataset["test"]))]
print(sample)
inputs = tokenizer(sample['prompt'], return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))