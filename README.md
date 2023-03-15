<p align="center">
<img src="static/images/banner.png" alt="drawing" width="600"/>
</p>

# InterroLang

TalkToModel (Slack et al., 2022) adaptation to NLP use cases (question answering, hate speech detection, dialogue act classification).


## Running with conda / virtualenv

Create the environment and install dependencies.

```shell
conda create -n ttm python=3.9
conda activate ttm
```

Install the requirements
```shell
pip install -r requirements.txt
```

Download datasets (only **BoolQ** for now!) by running
```python
python get_data.py
```

Download the model from [Hugging Face](https://huggingface.co/andi611/distilbert-base-uncased-qa-boolq/tree/main)  
and put it under _/configs_ as _./configs/boolq_model_.

You can launch the Flask web app via
```python
python flask_app.py
```


## Datasets / Use cases
* Question Answering (BoolQ)
* Hate Speech Detection (OLID)
* Dialogue Act Classification (DailyDialog)

## Explanation modules
* Feature Attribution (feature_importance)
* Counterfactual Generation (counterfactuals)
* Similar Examples (similarity)

## Data modules
* Filtering (utils)

