## Running with conda / virtualenv

Create the environment and install dependencies.

```shell
conda create -n ttm python=3.9
conda activate ttm
```

Create a directory to save the model and write log
eg

```shell
mkdir t5
```

Change the directory path on line 80 and 88 in t5_trainer.py

```shell
output_dir="/hd2/sahil/t5",
logging_dir=f"/hd2/sahil/t5/logs",
```

Run python command to train the model
```python
python t5_trainer.py
```

For inference,load the latest checkpoint from t5 in line 14 of t5_inference.py

```
model = AutoModelForSeq2SeqLM.from_pretrained("/hd2/sahil/t5/checkpoint-95")
```

run inference of one example to check the output
```shell
python t5_infernce.py
```