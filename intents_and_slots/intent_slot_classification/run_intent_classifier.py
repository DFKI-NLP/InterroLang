from transformers import AutoAdapterModel, AutoTokenizer
from transformers import TextClassificationPipeline

model_name = "bert-base-uncased"
adapter_name = "adapters/all"
model = AutoAdapterModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
intent_adapter = model.load_adapter(adapter_name)
model.load_head(adapter_name)


all_intents = ["include", "nlpcfe", "similar", "predict", "describe_self", "describe_data", "show", "likelihood", "describe_model", "describe_function", "score", "count_data", "label", "mistakes", "keywords", "nlpattribute", "rationalize", "global_topk", "stats"]
id2label_str = dict()
for i, intent_name in enumerate(all_intents):
    id2label_str[i] = intent_name

def get_sorted_labels(labels_dict):
    labels = []
    for entry in labels_dict:
        labels.append((id2label_str[int(entry["label"].replace("LABEL_",""))], entry["score"]))
    labels.sort(key=lambda x: x[1], reverse=True)
    return labels

intext = "Show me what does the model predict for id 67?"
intexts = ["Which samples have a word \"joy of abstraction\" and a token \"mathematics\" in them", "what is the prediction for id 14 in the dataset", "show me three counterfactuals examples for this instance", "what are the top two most frequent tokens in the development data?", "What are the predictions for id 7 and 45?", "what can you do for me?", "Show me all instances with the word spider"]
model.set_active_adapters(intent_adapter)
intent_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, task="all", device=-1)

intent_anno = intent_classifier(intexts)
for i, intext in enumerate(intexts):
    sorted_labels = get_sorted_labels(intent_anno[i])[:3]
    print(intext, sorted_labels)
