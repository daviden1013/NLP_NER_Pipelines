# NLP_NER_Pipelines
This is a general NER toolkit for BERT model training, evaluation, and prediction. The **BIO_pipeline** processes annotated corpora (with labels) and creates word-token level documents (.io or .bio) for training and evaluation. A split of development set and evaluation set should be considered. The **Training_pipeline** inputs the word-tokens (with labels) from development set, and train a BERT-familiy model. Checkpoints are saved to disk. The **Evaluation_pipeline** use a specified checkpoint to predict on the evaluation set, and perform entity-level evaluation. Both exact matching and partial matching with precision, recall, and F1 are reported. Once a final/ production NLP model is made and saved to disk, the **Prediction_pipeline** use it to do prediction. 

Framework: **PyTorch**, **Transformers**

![alt text](https://github.com/daviden1013/NLP_NER_Pipelines/blob/main/Pipelines%20diagram.png)
