# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import pprint
import yaml
import os
from transformers import BertTokenizer
from transformers import AutoModelForTokenClassification
import torch.optim as optim
from modules.Training_utilities import BIO_feeder, NER_Dataset
from modules.Prediction_utilities import Data_holder
from datetime import datetime

def main():
  print('Prediction pipeline starts:')
  print(datetime.now())
  """ load config """
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))
  
  print('Config loaded:')
  pprint.pprint(config)
  print(datetime.now())
  """ Load docudments """
  # Load documents into dict {docuemnt_id:text}
  # doc_dict = 
  print(f'Document loaded: {len(doc_dict)} documents')
  print(datetime.now())
  """ Load model """
  # model = 
  print('Model loaded')
  print(datetime.now())
  """ Make Data holder """
  label_map = config['label_map']
  holder = Data_holder(doc_dict, label_map)
  tokenizer = BertTokenizer.from_pretrained(config['tokenizer'])

  """ Make dataset """
  dataset = NER_Dataset(bios=holder.bio, 
                        tokenizer=tokenizer, 
                        label_map=label_map, 
                        word_seq_lenght=config['word_token_length'], 
                        step=config['word_token_length'],
                        token_seq_length=config['wordpiece_token_length'])
      
  """ Prediction """
  predictor = NER_Predictor(model=model,
                            tokenizer=tokenizer,
                            dataset=dataset,
                            label_map=label_map,
                            batch_size=config['batch_size'])
  token_pred_df = predictor.predict()
  
  """ Get predicted entity """
  entities = holder.Predict_to_entity(token_pred_df, mode=config['BIO_mode'])
