# prediction

### likelihood.txt
* 2x `likelihood [E]`
* 12x `filter id and likelihood [E]`
* 2x `includes {span} and likelihood [E]`

Action defined in prediction/prediction_likelihood.py

### mistakes.txt
* 5x `mistake sample [E]`
* 4x `mistake typical [E]`
* 9x `includes {span} and mistake sample [E]`
* 7x `includes {span} and mistake typical [E]`

Action defined in prediction/mistakes.py

### predict.txt
* 5x `predict [E]`
* 3x `predict [E]` (custom_input_prediction.txt)
* 10x `predict [E]` (custom_input_prediction_chatgpt.txt) [GPT-4 generated]
* 10x `filter id and predict [E]`

Action defined in prediction/predict.py


### score.txt
* 14x `score accuracy [E]`
* 1x `score accuracy f1 [E]`
* 2x `score default [E]`
* 3x `score f1 [E]`
* 1x `score f1 micro [E]`
* 1x `score f1 macro [E]`
* 1x `score f1 weighted [E]`
* 2x `score precision [E]`
* 1x `score precision micro [E]`
* 1x `score precision macro [E]`
* 1x `score precision weighted [E]`
* 3x `score recall [E]`
* 1x `score recall micro [E]`
* 1x `score recall macro [E]`
* 1x `score recall weighted [E]`
* 2x `score roc [E]`
* 3x `includes {span} and score accuracy [E]`
* 1x `includes {span} and score accuracy and score accuracy [E]`
* 1x `includes {span} and score precision [E]`
* 2x `includes {span} and score npv [E]`
* 2x `includes {span} and score ppv [E]`
* 1x `includes {span} and score sensitivity [E]`
* 1x `includes {span} and score specificity [E]`
* 2x `previousfilter and score npv [E]`
* 2x `previousfilter and score ppv [E]`
* 1x `previousfilter and score sensitivity [E]`
* 1x `previousfilter and score specificity [E]`

Action defined in prediction/score.py

