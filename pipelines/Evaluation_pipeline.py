# -*- coding: utf-8 -*-
import os
import re
import argparse
from easydict import EasyDict
import yaml
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from modules.Training_utilities import BIO_feeder, NER_Dataset, evaluate_entity
from modules.Prediction_utilities import NER_Predictor
from modules.Utilities import Tokens_to_entities
from datetime import datetime
import pprint

def main():
  print('Evaluation pipeline starts:')
  print(datetime.now())
  parser = argparse.ArgumentParser()
  add_arg = parser.add_argument
  add_arg("-c", "--config", help='path to config file', type=str)
  args = parser.parse_known_args()[0]
  
  with open(args.config) as yaml_file:
    config = EasyDict(yaml.safe_load(yaml_file))

  print('Config loaded:')
  pprint.pprint(config)
  print(datetime.now())
  
  label_map = config['label_map']
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  """ load test_id """
  with open(config['test_id_file']) as f:
    lines = f.readlines()
  test_id = [line.strip() for line in lines]
  feeder = BIO_feeder(config['BIO_dir'], test_id)
    
  dataset = NER_Dataset(feeder.get_BIO(), 
                        tokenizer, 
                        label_map, 
                        word_seq_lenght=config['word_token_length'], 
                        step=config['word_token_length'],
                        token_seq_length=config['wordpiece_token_length'])
    
  """ load model """
  best = AutoModelForTokenClassification.from_pretrained(config['base_model'], num_labels=len(label_map))
  
  if config['checkpoint'] == 'best':
    model_names = [f for f in os.listdir(config['checkpoint_dir']) if '.pth' in f]
    best_model_name = sorted(model_names, key=lambda x:int(re.search("-(.*?)_", x).group(1)))[-1]
    print(f'Evaluate model: {best_model_name}')
    print(datetime.now())
    best.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], best_model_name), 
                                    map_location=torch.device('cpu')))
  
  else:
    print(f"Evaluate model: {config['checkpoint']}")
    print(datetime.now())
    best.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], config['checkpoint']), 
                                    map_location=torch.device('cpu')))

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  best.to(device)
  best.eval()


  """ Prediction """
  predictor = NER_Predictor(model=best,
                            tokenizer=tokenizer,
                            dataset = dataset,
                            label_map=label_map,
                            batch_size=config['batch_size'])
  
  token_pred_df = predictor.predict()
  
  """ Convert predicted word-piece to word tokens """
  token_pred_df['pred'] = token_pred_df[[f'prob_{v}' for v in label_map.keys()]].\
    idxmax(axis=1).str.replace('prob_', '')
  token_pred_df['prob'] = token_pred_df[[v for v in token_pred_df.columns if v[:5] == 'prob_']].max(axis=1)
     
  token_pred_df = token_pred_df.sort_values(['document_id', 'start', 'end', 'prob']). \
    groupby(['document_id', 'start', 'end']).last()
     
  token_pred_df = token_pred_df[['pred', 'prob']]
  token_pred_df.reset_index(inplace=True)
  
  """ Get entities """
  entity_pred = Tokens_to_entities(token_df=token_pred_df, mode=config['BIO_mode'], 
                                   label_varname='pred', n_level=len(label_map))
  print('Token-level prediction made')
  print(datetime.now())
  """ Load Gold standard """
  gold = []
  for i in test_id:
    df = pd.read_csv(os.path.join(config['BIO_dir'], f"{i}.{config['BIO_mode']}"), 
                       encoding='utf-8', na_values=[''], keep_default_na=False)
    df['document_id'] = i
    gold.append(df)
  
  toekn_gold_df = pd.concat(gold)
  entity_gold = Tokens_to_entities(token_df=toekn_gold_df, mode=config['BIO_mode'], 
                                   label_varname='label', n_level=len(label_map))

  """ Output evaluation """
  eval_dir = os.path.join(config['out_path'], 'evaluations', config['run_name'])
  if not os.path.isdir(eval_dir):
    os.makedirs(eval_dir)
    
  evaluation = evaluate_entity(entity_pred, entity_gold)
  if config['output_predictions']:
    entity_pred.to_pickle(os.path.join(eval_dir, f"{config['run_name']} entity_pred.pickle"))
    entity_gold.to_pickle(os.path.join(eval_dir, f"{config['run_name']} entity_gold.pickle"))
  
  evaluation.to_csv(os.path.join(eval_dir, f"{config['run_name']} evaluation.csv"), index=False)
  

if __name__ == '__main__':
  main()