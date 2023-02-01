# -*- coding: utf-8 -*-
import torch
PATH = r'E:\David projects\ADE medication NER'

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
tokenizer.save_pretrained(os.path.join(PATH, 'Production Model'))

from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=19)
model.load_state_dict(torch.load(os.path.join(PATH, 'checkpoints', 'run1', 'Epoch-5_trainloss-0.0613_validloss-0.0869.pth'), 
                                    map_location=torch.device('cpu')))
model.save_pretrained(os.path.join(PATH, 'Production Model'))
