from datasets import load_dataset

boolq_val = load_dataset("super_glue", "boolq", split="validation").to_csv('data/boolq_validation.csv')
boolq_train = load_dataset("super_glue", "boolq", split="train").to_csv('data/boolq_train.csv')
