# Directory for fine-tuned parsing models

To test out the current model, download it from the link...

**FLAN-T5-base (250M)**  
https://cloud.dfki.de/owncloud/index.php/s/txzg9ZCdYqB39j9/download

...and put the folder (flan-t5-base) inside parsing/t5/models (this directory).

Afterwards, change `GlobalArgs.config` in *global_config.gin* in the root directory to the following:

```
GlobalArgs.config = "./configs/boolq_flan-t5.gin"
```
