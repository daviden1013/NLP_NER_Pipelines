# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import pprint
import yaml
import os
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch.optim as optim
from modules.Training_utilities import BIO_feeder, NER_Dataset, NER_Trainer
from datetime import datetime

def main():
  print('Training pipeline starts:')
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
  """ make training datasets """
  with open(config['deve_id_file']) as f:
    lines = f.readlines()
  dev_id = [line.strip() for line in lines]
  
  train_feeder = BIO_feeder(config['BIO_dir'], dev_id)
  train_feeder.train_valid_split(valid_ratio=float(config['valid_ratio']))
  
  train_bios, valid_bios = train_feeder.get_BIO()
  label_map = config['label_map']
  
  tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
  train_dataset = NER_Dataset(train_bios, 
                              tokenizer, 
                              label_map, 
                              word_seq_lenght=config['word_token_length'], 
                              step=config['slide_steps'],
                              token_seq_length=config['wordpiece_token_length'])
  
  valid_dataset = NER_Dataset(valid_bios, 
                              tokenizer, 
                              label_map, 
                              word_seq_lenght=config['word_token_length'], 
                              step=config['slide_steps'],
                              token_seq_length=config['wordpiece_token_length'])
  print('Training dataset created')
  print(datetime.now())
  
  """ define model """
  model = AutoModelForTokenClassification.from_pretrained(config['base_model'], num_labels=len(label_map))
  optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))
  
  print('Model loaded')
  print(datetime.now())
  """ Training """
  trainer = NER_Trainer(run_name=config['run_name'], 
                        model=model,
                        n_epochs=config['n_epochs'],
                        train_dataset=train_dataset,
                        batch_size=config['batch_size'],
                        optimizer=optimizer,
                        valid_dataset=valid_dataset,
                        save_model_mode='best',
                        save_model_path=os.path.join(config['out_path'], 'checkpoints'),
                        log_path=os.path.join(config['out_path'], 'logs'))
  
  trainer.train()

if __name__ == '__main__':
  main()