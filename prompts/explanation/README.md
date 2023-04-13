# explanation

### cfe.txt
* 17x `filter id and explain cfe [E]`
* 1x `filter id and predict and explain cfe [E]`
* 6x `filter id and explain features and explain cfe [E]`
* 1x `filter id and explain cfe and explain features and predict [E]`

Action defined in explanation/cfe_generation.py


### global_feature_importance.txt
* 21x `important all [E]`
* 9x `important {classname} [E]`
* 11x `important topk [E]`
* 19x `includes {span} and important all [E]`
* 3x `includes {span} and important topk [E]`

Action defined in explanation/topk.py

### local_feature_importance.txt
* 10x `filter id and nlpattribute all [E]`
* 5x `filter id and predict and nlpattribute all [E]`
* 6x `filter id and nlpattribute topk [E]`

**GPT-4 generated**
* 6x `filter id and nlpattribute all [E]`
* 9x `filter id and nlpattribute topk [E]`
* 3x `filter id or filter id and nlpattribute topk [E]`
* 2x `filter id or filter id or filter id and nlpattribute topk [E]`

Action defined in explanation/feature_importance.py

### rationalize.txt
* 4x `filter id and rationalize [E]`

**GPT-4 generated**
* 20x `filter id and rationalize [E]`

Action defined in explanation/rationalize.py
