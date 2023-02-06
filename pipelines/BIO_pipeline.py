# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import yaml
from modules.Training_utilities import Label_studio_BIO_converter
from datetime import datetime
import pprint

  
def main():
  print('BIO pipeline starts:')
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
  converter = Label_studio_BIO_converter(ann_file=config['ann_file'],
                                        BIO_dir=config['BIO_dir'],
                                        mode=config['BIO_mode'])
  
  converter.pop_BIO()

if __name__ == '__main__':
  main()