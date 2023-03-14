from datasets import load_dataset
boolq = load_dataset("super_glue", "boolq", split="validation").to_csv('data/boolq_validation.csv')
boolq = load_dataset("super_glue", "boolq", split="train").to_csv('data/boolq_train.csv')

