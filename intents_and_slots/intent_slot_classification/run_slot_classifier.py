from transformers import TokenClassificationPipeline
from transformers import AutoAdapterModel, AutoTokenizer
device = -1

model_name = "bert-base-uncased"
adapter_name = "adapters/slots"
tagger_model = AutoAdapterModel.from_pretrained(model_name)
adapter_tokenizer = AutoTokenizer.from_pretrained(model_name)
slots_adapter = tagger_model.load_adapter(adapter_name)
tagger_model.load_head(adapter_name)
tagger_model.set_active_adapters([slots_adapter])

def get_slot_annotations(text_anno, intext):
    intext_chars = list(intext)
    slot_types = ["class_names", "data_type", "id", "includetoken", "metric", "number"]
    slot2spans = dict()
    for anno in text_anno:
        slot_type = anno["entity"][2:]
        if not(slot_type) in slot2spans:
            slot2spans[slot_type] = []
        slot2spans[slot_type].append((anno["word"], anno["start"], anno["end"], anno["entity"]))
    final_slot2spans = dict()
    for slot_type in slot2spans:
        final_slot2spans[slot_type] = []
        span_starts = [s for s in slot2spans[slot_type] if s[-1].startswith("B-")]
        span_starts.sort(key=lambda x: x[1])
        span_ends = [s for s in slot2spans[slot_type] if s[-1].startswith("I-")]
        span_ends.sort(key=lambda x: x[1])
        for i, span_start in enumerate(span_starts):
            if i<len(span_starts)-1:
                next_span_start = span_starts[i+1]
            else:
                next_span_start = None
            selected_ends = [s[2] for s in span_ends if s[1]>=span_start[1] and (next_span_start is None or s[1]<next_span_start[1])]
            if len(selected_ends)>0:
                span_end = max(selected_ends)
            else:
                span_end = span_start[2]
            span_start = span_start[1]
            final_slot2spans[slot_type].append("".join(intext_chars[span_start:span_end]))
    return final_slot2spans



### working version ###

intexts = ["Which samples have a word \"joy of abstraction\" and a token \"mathematics\" in them", "what is the prediction for id 14 in the dataset", "show me three counterfactuals examples for this instance", "what are the top two most frequent tokens in the development data?", "What are the predictions for id 7 and 45?", "what can you do for me?"]
slots_tagger = TokenClassificationPipeline(model=tagger_model, tokenizer=adapter_tokenizer, task="slots", device=device)
print(slots_tagger)
slot_anno = slots_tagger(intexts)
for i, intext in enumerate(intexts):
    print(intext, slot_anno[i])
    slot2spans = get_slot_annotations(slot_anno[i], intext)
    for slot in slot2spans:
        print(slot, slot2spans[slot])

