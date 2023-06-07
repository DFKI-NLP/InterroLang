<p align="center">
<img src="static/images/banner.png" alt="drawing" width="600"/>
</p>


[![Python application](https://github.com/nfelnlp/InterroLang/actions/workflows/github.yml/badge.svg)](https://github.com/nfelnlp/InterroLang/actions/workflows/github.yml/badge.svg)

# InterroLang

A TalkToModel (Slack et al., 2022) adaptation to NLP use cases (question answering, hate speech detection, dialogue act classification).
The name is a word-play on [Interrobang](https://en.wikipedia.org/wiki/Interrobang), a ligature of question mark and exclamation mark, and **Lang**uage models.
Our tool offers a dialogue-based exploration of NLP interpretability methods (feature attribution, counterfactuals and perturbations, free-text rationalization) and dataset analyses (similar examples, keywords, label distribution).

<p align="center">
&nbsp;&nbsp;&nbsp;
<img src="static/images/interrolang.gif" alt="drawing" width="600"/>
</p>

## InterroLang Interface
![](./static/images/interface.png)
We consider 7 categories of operations.
- About: Capabilities of our system.
- Metadata: Information in terms of data, model, labels.
- Prediction: Various operations related to gold labels and model predictions.
- Understanding: Keyword-based analysis and retrieval of similar instances.
- Explanation: Feature attribution methods (local-, class-, global-level) and free-text rationalization.
- Perturbation: Methods to change some parts of an instance, e.g. such that the label of the instance would change.
- Custom input: We not only allow instances from the dataset but also instances given by users. (More details see [below](https://github.com/nfelnlp/InterroLang#process))

### Dataset Viewer
We provide a dataset view with which users can explore instances contained in the pre-defined dataset (in screenshot, BoolQ dataset is used). Users can search instances that include the entered string. 
![](./static/images/data_viewer.png)

## Datasets / Use cases
* Question Answering ([BoolQ](https://huggingface.co/datasets/boolq))
* Hate Speech Detection ([OLID](https://scholar.harvard.edu/malmasi/olid))
* Dialogue Act Classification ([DailyDialog](https://huggingface.co/datasets/daily_dialog))


## Running with conda / virtualenv

### Create the environment and install dependencies.

#### Conda
```shell
conda create -n interrolang python=3.9
conda activate interrolang
```

#### venv
```shell
python -m venv venv
source venv/venv/activate
```

### Install the requirements
```shell
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download omw-1.4
python -m nltk.downloader omw-1.4

# Punkt
python -m nltk.downloader punkt

# Install spacy
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
Install [polyjuice-nlp](https://github.com/tongshuangwu/polyjuice) and its dependencies due to some [issues](https://github.com/tongshuangwu/polyjuice/issues/10) from polyjuice:
```bash
cd utils
bash dependency.sh
```

### Download models
In our tool, we currently use the following Transformer models: 

- DistilBERT fine-tuned on BoolQ: https://huggingface.co/andi611/distilbert-base-uncased-qa-boolq (HF)
- MultiBERT fine-tuned on OLID: https://huggingface.co/sinhala-nlp/mbert-olid-en (HF)
- BERT fine-tuned on DailyDialog: https://cloud.dfki.de/owncloud/index.php/s/m72HGNLW2TyCABr (DFKI)

#### For BoolQ and OLID Model:
Put them under `./data` and name the folders `boolq_model` and `olid_model` respectively.

#### For Daily Dialog model:
Put the file `5e_5e-06lr` under `./explained_models/da_classifier/saved_model`


### Set up configuration
In `./configs`, there are all gin config files for all three datasets with different parsing models. You can choose one of them and set its path in `./global_config.gin`:
```
GlobalArgs.config = "./configs/boolq_adapter.gin"
```

### Run the application
You can launch the Flask web app via
```python
python flask_app.py
```


## Running with Docker

If you want to run with Docker, you can build the docker app
```bash
sudo docker build -t interrolang .
```

And then run the image
```bash
sudo docker run -d -p 4000:4000 interrolang
```

## User Guide
After the project is set up, we provide a [user guidance](https://github.com/nfelnlp/InterroLang/blob/main/utils/user_guide.py) using Selenium to demonstrate how to use our system if you have Chrome Browser available.

## How to use custom input
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

