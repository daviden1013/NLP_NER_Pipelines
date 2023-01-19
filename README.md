# NLP_NER_Pipelines
This is a general NER toolkit for BERT model training, evaluation, and prediction. The **BIO_pipeline** processes annotated corpora (with labels) and creates word-token level documents (.io or .bio) for training and evaluation. A split of development set and evaluation set should be considered. The **Training_pipeline** inputs the word-tokens (with labels) from development set, and train a BERT-familiy model. Checkpoints are saved to disk. The **Evaluation_pipeline** use a specified checkpoint to predict on the evaluation set, and perform entity-level evaluation. Both exact matching and partial matching with precision, recall, and F1 are reported. Once a final/ production NLP model is made and saved to disk, the **Prediction_pipeline** use it to do prediction. 

Framework: **PyTorch**, **Transformers**

Annotation tool: **MAE**

![alt text](https://github.com/daviden1013/NLP_NER_Pipelines/blob/main/Pipelines%20diagram.png)

## Demo
Prepare project file system. We create folders as below (all modules under "pipelines" can be cloned from this repository):
```
Project folder
  - BIO
  - train_test_id
  - pipelines
    - BIO_pipeline.py
    - Training_pipeline.py
    - Evaluation_pipeline.py
    - Prediction_pipeline.py
    - configs
      - configs_run_0.yaml
    - modules
      - __init__.py
      - Prediction_utilities.py
      - Training_utilities.py
      - Utilities.py
```
Edit config file. All the pipelines use the same config file. 

The parameters for word-token (IO/BIO) creation are:
```
########################
# BIO processing parameters
########################
# directory for annotated XML files
XML_dir: [XML_dir:str]
# directory for IO/BIO files
BIO_dir: [BIO_dir:str]
# mode, either IO or BIO
BIO_mode: [BIO_mode:str]
```
The output from MAE annotation tool is in .xml format. The XML_dir specifies a directory with all the .xml files. If different annotation tool was used, modify the children class in BIO_pipeline. The BIO_dir specifies the output directory for .io or .bio files. BIO_mode specifies whether the labels will include "B-" and "I-" (BIO) or without (IO). When 1) training data is sufficient or 2) same type of entities could show up back-to-back, BIO is suggested. Otherwise IO could be more efficient (reduces the categories for prediction by half). Once executed, the **BIO_pipeline** creates .io or .bio files in BIO_dir. 

The parameters for NLP model fine-tuning are:
```
########################
# Model fine-tune parameters
########################
# development set file path
deve_id_file: [deve_id_path:str]
# ratio of validation set. Will be sampled from development set
valid_ratio: [ratio:float]
# Define entity labels and category numbers
label_map: 
  O: 0
  label_1: 1
  label_2: 2
# tokenizer path
tokenizer: [tokenizer:str]
# Word tokens to be included in a segment
word_token_length: [length:int]
# step of sliding window for training instance creation
slide_steps: [steps:int]
# wordpiece tokens to include in a training instance
wordpiece_token_length: [length:int]
# base NLP model file path
base_model: [model_path:str]
# learning rate
lr: [rate:float]
# n_ephoch
n_epochs: [N:int]
# batch_size
batch_size: [size:int]
# Output path
out_path: [path:str]
```
deve_id_file specifies the id list (per document_id per row) for development set. The pipeline will use the ids to pull .io or .bio files. Training and validation split is handled automatically, just specify the ratio in valid_ratio. label_map defines the entity types and a numeric code. "O" means other (non-entity). word_token_length specifies the number of word-tokens to include in a training instance. slide_steps speficies the number of word-tokens to slide to create the next training instance. wordpiece_token_length defines the exact number of tokens input into BERT. Padding and truncating applies. out_path specifies the root folder for checkpoints and logs. The **Training_pipeline** creates the following files:
```
out_path
  - checkpoints
    - [run name 1]
      - checkpoints 1
      - checkpoints 2
      ...
    - [run name 2]
    ...
  - logs
    - [run name 1]
      - log
      ...
```
The parameters for NLP model evaluation are:
```
########################
# Evaluation parameters
########################
# test set file path
test_id_file: [eval_id_path:str]
# checkpoint path
checkpoint_dir: [checkpoint path:str]
# checkpoint filename to evaluate, or "best" to pull the checkpoint with min validation loss.
checkpoint: [checkpoint file:str]
```
test_id_file specifies a list of document_id in evaluation set. The pipeline will pull them from BIO_dir. checkpoint_dir specifies the directory with checkpoints. checkpoint could be the filename of checkpoint, or "best". 

Once the config file is completed, run:
```
$cd [Project folder]
$python BIO_pipeline.py -c ./configs/configs_run_0.yaml
$python Training_pipeline.py -c ./configs/configs_run_0.yaml
$python Evaluation_pipeline.py -c ./configs/configs_run_0.yaml
```

The **Prediction_pipeline** provides a coding template for production. The input should be a dictionary:
```
{[document_id 0] : [text 0],
[document_id 1] : [text 1],
[document_id 2] : [text 2]
...}
```
