# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict
import yaml
from typing import Tuple
import os
from modules.Training_utilities import BIO_converter
from datetime import datetime
from tqdm import tqdm
import csv
import pprint

""" Define annotation file parsing method """
class BRAT_BIO_converter(BIO_converter):
  def __init__(self, txt_dir:str, ann_dir:str, BIO_dir:str, mode:str):
    """
    This class inputs a directory with annotation files, outputs BIOs

    Parameters
    ----------
    txt_dir: str
      Directory of text files
    ann_dir : str
      Directory of annotation files
    BIO_dir : str
      Directory of BIO files 
    mode : str
      choice of {'BIO', 'IO'} for output
    """
    self.txt_dir = txt_dir
    self.ann_dir = ann_dir
    self.BIO_dir = BIO_dir
    assert mode in {'BIO', 'IO'}, "mode must be one of {'BIO', 'IO'}"
    self.mode = mode
  
  def parse_annotation(self, txt_filename:str, ann_filename:str) -> Tuple[str, list]:
    with open(os.path.join(self.txt_dir, txt_filename), 'r') as f:
      text = f.read()
        
    ann_list = []
    with open(os.path.join(self.ann_dir, ann_filename), 'r') as f:
      lines = f.readlines()
    for line in lines:
      if line[0] == 'T':
        l = line.split()
        tag_id = l[0]
        tag_name = l[1]
        start = int(l[2])
        i = 3
        while True:
          if ';' in l[i]:
            i += 1
          else:
            end = int(l[i])
            break
            
        ann_list.append((tag_id, tag_name, start, end))
        
    return text, ann_list
  
  
  def pop_BIO(self):
    """
    This method iterate through annotation files and create BIO
    """
    txt_files = sorted([f for f in os.listdir(self.txt_dir) 
                 if os.path.isfile(os.path.join(self.txt_dir, f)) and f[-4:] == '.txt'])
    ann_files = sorted([f for f in os.listdir(self.ann_dir) 
                 if os.path.isfile(os.path.join(self.ann_dir, f)) and f[-4:] == '.ann'])
    loop = tqdm(zip(txt_files, ann_files), total=len(ann_files), leave=True)
    for txt_file, ann_file in loop:
      txt, ann = self.parse_annotation(txt_file, ann_file)
      
      if self.mode == 'BIO':
        bio_list = self._get_BIO(txt, ann)
        filename = ann_file.replace('.ann', '.bio')
      else:
        bio_list = self._get_IO(txt, ann)
        filename = ann_file.replace('.ann', '.io')
        
      with open(os.path.join(self.BIO_dir, filename), 'w', newline='', encoding='utf-8') as file:
        csv_out=csv.writer(file)
        csv_out.writerow(['token','start','end','label'])
        for row in bio_list:
          csv_out.writerow(row)
   
  
""" Pipeline """
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
  converter = BRAT_BIO_converter(txt_dir=config['txt_dir'],
                                 ann_dir=config['ann_dir'],
                                BIO_dir=config['BIO_dir'],
                                mode=config['BIO_mode'])
  
  converter.pop_BIO()

if __name__ == '__main__':
  main()