# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import pprint
import yaml
import os
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch.optim as optim
from modules.Training_utilities import BIO_feeder, NER_Dataset
from modules.Prediction_utilities import Data_holder, NER_Predictor
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
  doc_dict = {}
  for file in os.listdir(config['predict_doc_dir']):
    with open(os.path.join(config['predict_doc_dir'], file)) as f:
      txt = f.read()
    filename = file.replace('.txt', '')
    doc_dict[filename] = txt

  print(f'Documents loaded: {len(doc_dict)} documents')
  print(datetime.now())
  """ Make Data holder """
  label_map = config['label_map']
  holder = Data_holder(doc_dict, label_map)
  tokenizer = AutoTokenizer.from_pretrained(config['predict_model'])
  
  """ Load model """
  model = AutoModelForTokenClassification.from_pretrained(config['predict_model'], num_labels=len(label_map))
  print('Model loaded')
  print(datetime.now())
  
  """ Make dataset """
  dataset = NER_Dataset(bios=holder.bio, 
                        tokenizer=tokenizer, 
                        label_map=label_map, 
                        word_seq_lenght=config['word_token_length'], 
                        step=config['word_token_length'],
                        token_seq_length=config['wordpiece_token_length'],
                        has_label=False)
      
  """ Prediction """
  predictor = NER_Predictor(model=model,
                            tokenizer=tokenizer,
                            dataset=dataset,
                            label_map=label_map,
                            batch_size=config['batch_size'])
  token_pred_df = predictor.predict()
  
  """ Get predicted entity """
  entities = holder.Predict_to_entity(token_pred_df, mode=config['BIO_mode'])
  entities['entity_id'] = entities.apply(lambda x:f'{x.document_id}_{x.start}_{x.end}', axis=1)

  """ Save """
  entities.to_pickle(config['predict_outfile'])
  
if __name__ == '__main__':
  main()
