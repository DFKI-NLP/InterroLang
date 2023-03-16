# prediction

### likelihood.txt
* 2x `likelihood [E]`
* 12x `filter id and likelihood [E]`
* 2x `{filter_parse} and likelihood [E]`

Action defined in prediction/prediction_likelihood.py

### mistakes.txt
* 5x `mistake sample [E]`
* 4x `mistake typical [E]`
* 9x `{filter_parse} and mistake sample [E]`
* 7x `{filter_parse} and mistake typical [E]`

Action defined in prediction/mistakes.py

### predict.txt
* 5x `predict [E]`
* 10x `filter id and predict [E]`
* 1x `{filter_parse} and predict [E]` 

Action defined in prediction/predict.py

### score.txt
* 14x `score accuracy [E]`
* 1x `score accuracy f1 [E]`
* 2x `score default [E]`
* 3x `score f1 [E]`
* 2x `score precision [E]`
* 3x `score recall [E]`
* 2x `score roc [E]`
* 3x `{filter_parse} and score accuracy [E]`
* 1x `{filter_parse} and score accuracy and score accuracy [E]`
* 1x `{filter_parse} and score precision [E]`
* 2x `{filter_parse} and score npv [E]`
* 2x `{filter_parse} and score ppv [E]`
* 1x `{filter_parse} and score sensitivity [E]`
* 1x `{filter_parse} and score specificity [E]`
* 2x `previousfilter and score npv [E]`
* 2x `previousfilter and score ppv [E]`
* 1x `previousfilter and score sensitivity [E]`
* 1x `previousfilter and score specificity [E]`

Action defined in prediction/score.py

