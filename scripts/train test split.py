# -*- coding: utf-8 -*-
PATH = r'E:\David projects\ADE medication NER'
import os

train = [f.replace('.ann', '') for f in os.listdir(os.path.join(PATH, 'data', 'training_20180910')) if f[-4:] == '.ann']
test = [f.replace('.ann', '') for f in os.listdir(os.path.join(PATH, 'data', 'test')) if f[-4:] == '.ann']

with open(os.path.join(PATH, 'train_test_id', 'train_id'), 'w') as f:
  for l in train:
    f.write("%s\n" % l)
    
with open(os.path.join(PATH, 'train_test_id', 'test_id'), 'w') as f:
  for l in test:
    f.write("%s\n" % l)