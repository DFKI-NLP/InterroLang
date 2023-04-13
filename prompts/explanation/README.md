# explanation

### cfe.txt
* 17x `filter id and explain cfe [E]`
* 1x `filter id and predict and explain cfe [E]`
* 6x `filter id and explain features and explain cfe [E]`
* 1x `filter id and explain cfe and explain features and predict [E]`

Action defined in explanation/cfe_generation.py (WIP)

### explain.txt
* 10x `explain features [E]`
* 8x `filter id and explain features [E]`
* 2x `filter id and explain features and predict [E]`
* 3x `filter id and predict and explain features [E]`
* 20x `{filter_parse} and explain features [E]`

???

### global_feature_importance.txt
* 16x `important all [E]`
* 11x `important topk [E]`
* 1x `{filter_parse} and important all [E]`
* 3x `{filter_parse} and important topk [E]`

Action defined in explanation/topk.py

### local_feature_importance.txt
* 2x `filter id and nlpattribute all [E]`
* 6x `filter id and nlpattribute topk [E]`

### local_feature_importance_chatgpt.txt
* 6x `filter id and nlpattribute all [E]`
* 9x `filter id and nlpattribute topk [E]`
* 3x `filter id or filter id and nlpattribute topk [E]`
* 2x `filter id or filter id or filter id and nlpattribute topk [E]`

Action defined in explanation/feature_importance.py

### rationalize.txt
* 4x `filter id and rationalize [E]`

### rationalize_chatgpt.txt
* 20x `filter id and rationalize [E]`

Action defined in explanation/rationalize.py
