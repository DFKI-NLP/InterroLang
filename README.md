<p align="center">
<img src="static/images/banner.png" alt="drawing" width="600"/>
</p>

# InterroLang

TalkToModel (Slack et al., 2022) adaptation to NLP use cases (question answering, hate speech detection, dialogue act classification).

## InterroLang Interface
![](./static/images/interface.png)
We consider 7 categories of operations.
- About: Capabilities of our system.
- Metadata: information in terms of data, model, label.
- Prediction: various operations related to golden labels and predictions.
- Understanding: find some similar instances.
- Explanation: different kinds of methods to explain an instance / dataset.
- Custom input: we not only allow instances from the dataset but also instances given by user. (More details see below)
- Perturbation: operations would change some parts of instance such that the label of the instance would change.  

### Dataset Viewer
We provide a dataset view, with which user can explore instances containing in the current dataset (in screenshot, BoolQ dataset is used). User can enter a token to search instances that include the entered token. 
![](./static/images/data_viewer.png)

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

## How to get used models?
- BoolQ model: https://huggingface.co/andi611/distilbert-base-uncased-qa-boolq
- OLID model: https://huggingface.co/sinhala-nlp/mbert-olid-en
- Daily Dialog model: https://cloud.dfki.de/owncloud/index.php/s/m72HGNLW2TyCABr

## How to use custom input?
### Supported operations
1. feature importance on token level
2. feature importance on sentence level
3. prediction
4. similarity
5. rationalization

### Process
#### 1. Enter your custom input in the text area and then click send button. Be aware: you have to choose **"Custom input"** in the selection box.
![](./static/images/custom_input.png)

#### 2. After clicking the button, you could see your custom input in the terminal
![](./static/images/enter_custom_input.png)

#### 3. Then you should enter prompts for operations mentioned above. Operations that support custom input are highlighted with yellow border.
![](./static/images/buttons.png)

#### 4. In the end, click the send button and you will get the result. (In this example, we show the result of feature importance on token level on given custom input)
![](./static/images/result.png)

