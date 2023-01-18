# -*- coding: utf-8 -*-
import abc
from typing import List, Tuple, Dict
import os
import numpy as np
import pandas as pd
import string
import xml.etree.ElementTree as ET
import csv
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForTokenClassification
from modules.Utilities import word_tokenize
from tqdm import tqdm


class XML_BIO_converter:
  def __init__(self, xml_dir:str, BIO_dir:str, mode:str):
    """
    This class inputs a directory with XML files, outputs BIOs

    Parameters
    ----------
    xml_dir : str
      Directory of XML files
    BIO_dir : str
      Directory of BIO files 
    mode : str
      choice of {'BIO', 'IO'} for output
    """
    self.xml_dir = xml_dir
    self.BIO_dir = BIO_dir
    assert mode in {'BIO', 'IO'}, "mode must be one of {'BIO', 'IO'}"
    self.mode = mode
    
  def _tokenize(self, text:str) -> List:
    """
    This method inputs a text and split it by space and any punctuations
    Outputs a list of 3-tuple (word, start, end)

    Parameters
    ----------
    text : str
      Document content to be tokenized

    Returns
    -------
    out : List of 3-tuple
      a list of 3-tuple (word, start, end)
    """
    return word_tokenize(text)
    
  
  @abc.abstractmethod
  def _parse_XML(self, xml_filename:str) -> Tuple[str, list]:
    """
    This method inputs a xml_filename with dir
    outputs text content + list of tags 4-tuple 
    (tag_id, tag_name, start, end)

    Parameters
    ----------
    xml_filename : str
      xml filename with dir.
    """
    return NotImplemented
  

  def _get_BIO(self, xml_filename:str) -> List:
    """
    This method inputs a xml filename + dir
    outputs a BIO list
    it returns a list of 4-tuple (word, start, end, label)

    Parameters
    ----------
    xml_filename : str
      xml filename with dir.
    """
    text, tags = self._parse_XML(xml_filename)
    tokens = self._tokenize(text)
    out = []
    begin = True
    cur_tag_id = ''
    for token in tokens:
      label = 'O'
      tag_id = ''
      for tag in tags:
        if not (token[2] <= tag[2] or token[1] >= tag[3]): # word token overlap with tag
          tag_id = tag[0]
          if cur_tag_id != tag_id:
            begin = True
            cur_tag_id = ''
          
          if begin:
            cur_tag_id = tag_id
            label = 'B-' + tag[1]
            begin = False
          else:
            label = 'I-' + tag[1]

      out.append((token[0], token[1], token[2], label))

    return out
  
  
  def _get_IO(self, xml_filename:str) -> List:
    """
    This method inputs a xml filename + dir
    outputs a IO list
    it returns a list of 4-tuple (word, start, end, label)

    Parameters
    ----------
    xml_filename : str
      xml filename with dir.
    """
    text, tags = self._parse_XML(xml_filename)
    tokens = self._tokenize(text)
    out = []
    for token in tokens:
      label = 'O'
      for tag in tags:
        if not (token[2] <= tag[2] or token[1] >= tag[3]): # word token overlap with tag
          label = tag[1]

      out.append((token[0], token[1], token[2], label))

    return out
  
  
  def pop_BIO(self):
    """
    This method iterate through all xml files and output BIOs as csv format (.bio)
    """
    xml_files = [f for f in os.listdir(self.xml_dir) 
                 if os.path.isfile(os.path.join(self.xml_dir, f)) and f[-4:] == '.xml']
    loop = tqdm(xml_files, total=len(xml_files), leave=True)
    for xml in loop:
      if self.mode == 'BIO':
        bio_list = self._get_BIO(os.path.join(self.xml_dir, xml))
        filename = xml.replace('.xml', '.bio')
      else:
        bio_list = self._get_IO(os.path.join(self.xml_dir, xml))
        filename = xml.replace('.xml', '.io')
        
      with open(os.path.join(self.BIO_dir, filename), 'w', newline='', encoding='utf-8') as file:
        csv_out=csv.writer(file)
        csv_out.writerow(['token','start','end','label'])
        for row in bio_list:
          csv_out.writerow(row)


class BIO_feeder:
  def __init__(self, BIO_dir:str, id_list: List):
    """
    This class inputs a BIO directory that contains all BIO files, and a list of 
    file ids (for trainset and testset). Returns a Dict of {file id: bio contain string}

    Parameters
    ----------
    BIO_dir : str
      a folder that contains all BIO files
    id_list : List
      a Dict of {file id: bio contain string}
    """
    self.BIO_dir = BIO_dir
    self.id_list = id_list
    self._train_valid_splited = False
    self.train_id = None
    self.valid_id = None
    """ Decide mode BIO or IO based on first file in BIO_dir """
    f = os.listdir(self.BIO_dir)[0]
    self.mode = f.split('.')[-1]
    
  def train_valid_split(self, valid_ratio:float=0.2, seed:int=123):
    """
    This method samples from development set and creates train and validation sets
    as members. This method should only be called once.

    Parameters
    ----------
    valid_ratio : float, optional
      The ratio of validation set. The default is 0.2.
    seed : int, optional
      Random seed. The default is 123
    """
    assert self._train_valid_splited == False, \
    "_train_valid_splited is True. Make sure this is the first time calling this method."
    np.random.seed(seed)
    self.valid_id = np.random.choice(self.id_list, int(len(self.id_list) * valid_ratio), replace=False).tolist()
    self.train_id = [i for i in self.id_list if i not in self.valid_id]
    self._train_valid_splited = True

  def get_BIO(self) -> Dict:
    """
    Output Dict of {file id: bio contain string}.
    If _train_valid_splited == True, return tuple (train BIO, valid BIO)
    else reutrn train BIO only
    """
    
    """ Train-valid splited, return train BIO and valid BIO """
    if self._train_valid_splited:
      train_bio = {}
      for i in self.train_id:
        df = pd.read_csv(os.path.join(self.BIO_dir, f"{i}.{self.mode}"), 
                              encoding='utf-8', na_values=[''], keep_default_na=False)
        train_bio[i] = list(df.to_records(index=False))
        
      valid_bio = {}
      for i in self.valid_id:
        df = pd.read_csv(os.path.join(self.BIO_dir, f"{i}.{self.mode}"), 
                         encoding='utf-8', na_values=[''], keep_default_na=False)
        valid_bio[i] = list(df.to_records(index=False))
        
      return train_bio, valid_bio
        

    """ Return single BIO """
    bios = {}
    for i in self.id_list:
      df = pd.read_csv(os.path.join(self.BIO_dir, f"{i}.{self.mode}"), 
                       encoding='utf-8', na_values=[''], keep_default_na=False)
      bios[i] = list(df.to_records(index=False))
      
    return bios
    

class NER_Dataset(Dataset):
  def __init__(self, 
               bios: Dict, 
               tokenizer: BertTokenizer, 
               label_map: Dict,
               word_seq_lenght: int=32, 
               step: int=10,
               token_seq_length: int=64,
               has_label: bool=True):
    """
    This class inputs a dictionary of bio files as Dict {document_id: bio as List (tuple)}
    Output dict {input_ids, attention_mask, labels, start, end, document_id}

    Parameters
    ----------
    bios : Dict
      key=document_id, val=List(tuple) of bios. Must include columns: 
      TOKEN, START, END in correct order. LABEL is optional.
    tokenizer : BertTokenizer
      tokenizer
    label_map : Dict
      key=BIO tag, val=categorical code
    word_seq_lenght : int, optional
      Length of words in a segment. The default is 32.
    token_seq_length : int, optional
      Length of wordPiece tokens in a segment. The default is 64.
    has_label : bool, optional
      Specify if label exists in bio. True for training; false for prediction. 
      The default is True.
    """
    self.tokenizer = tokenizer
    self.label_map = label_map
    self.word_seq_lenght = word_seq_lenght
    self.step = step
    self.token_seq_length = token_seq_length
    self.has_label = has_label
    
    self.word_seq = []
    self.document_ids = []
    for document_id, bio in bios.items():        
      i = 0
      while True:
        self.word_seq.append(bio[i:i+self.word_seq_lenght])
        self.document_ids.append(document_id)
        if i >= len(bio):
          break
        i += self.step
        
  def __len__(self):
    """ 
    Return total number of instances (list of tuple(word-token, start, end, label))
    Each instance has length = word_seq_lenght
    """
    return len(self.word_seq)
  
  def __getitem__(self, idx):
    """
    Output dict {document_id, input_ids, attention_mask, start, end, (labels)}
    """
    word_seq = self.word_seq[idx]
    other_label = self.label_map['O']
    out = {'input_ids':[],
           'attention_mask':[],
           'start':[],
           'end':[]}
    if self.has_label:
      out['labels'] = []
    
    for word in word_seq:
      tokens = self.tokenizer.tokenize(word[0])
      if len(tokens) == 0:
        continue
      input_ids = self.tokenizer.encode(tokens, add_special_tokens=False)
      
      out['input_ids'].extend(input_ids)
      out['attention_mask'].extend([1]*len(tokens))
      out['start'].extend([word[1]]*len(tokens))
      out['end'].extend([word[2]]*len(tokens))
      if self.has_label:
        label_code = self.label_map[word[3]] if word[3] in self.label_map else other_label
        out['labels'].extend([label_code]*len(tokens))
      
    # truncate or padding to make token lenght = self.token_seq_length
    if len(out['input_ids']) > self.token_seq_length:
      for k in out.keys():
        out[k] = out[k][0:self.token_seq_length]
      
    else:
      for _ in range(self.token_seq_length - len(out['input_ids'])):
        out['input_ids'].append(0)
        out['attention_mask'].append(0)
        if self.has_label:
          out['labels'].append(other_label)
        out['start'].append(-1)
        out['end'].append(-1)
            
    out['document_id'] = [self.document_ids[idx]] * len(out['input_ids'])
    
    # make input_ids, attention_mask, (labels), start, end tensor
    for v in out.keys():
      if v != 'document_id':
        out[v] = torch.tensor(out[v])
      
    return out


class NER_Trainer():
  def __init__(self, run_name: str, model, n_epochs: int, train_dataset: Dataset, 
               batch_size: int, optimizer, 
               valid_dataset: Dataset=None, shuffle: bool=True, drop_last: bool=True,
               save_model_mode: str=None, save_model_path: str=None, log_path: str=None):

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.run_name = run_name
    self.model = model
    self.model.to(self.device)
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.valid_dataset = valid_dataset
    self.shuffle = shuffle
    self.save_model_mode = save_model_mode
    self.save_model_path = os.path.join(save_model_path, self.run_name)
    if save_model_path != None and not os.path.isdir(self.save_model_path):
      os.makedirs(self.save_model_path)
    self.best_loss = float('inf')
    self.train_dataset = train_dataset
    self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                   shuffle=self.shuffle, drop_last=drop_last)
    if valid_dataset != None:
      self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, 
                                     shuffle=False, drop_last=drop_last)
    else:
      self.valid_loader = None
    
    self.log_path = os.path.join(log_path, self.run_name)
    if log_path != None and not os.path.isdir(self.log_path):
      os.makedirs(self.log_path)
    self.tensorboard_writer = SummaryWriter(self.log_path) if log_path != None else None
    self.global_step = 0
    
  def evaluate(self):
    with torch.no_grad():
      valid_total_loss = 0
      for valid_batch in self.valid_loader:
        valid_input_ids = valid_batch['input_ids'].to(self.device)
        valid_attention_mask = valid_batch['attention_mask'].to(self.device)
        valid_labels = valid_batch['labels'].to(self.device)
        output = self.model(input_ids=valid_input_ids, 
                            attention_mask=valid_attention_mask, 
                            labels=valid_labels)
        valid_loss = output.loss
        valid_total_loss += valid_loss.item()
      return valid_total_loss/ len(self.valid_loader)
    
  def train(self):
    for epoch in range(self.n_epochs):
      train_total_loss = 0
      valid_mean_loss = None
      loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
      
      for batch_id, train_batch in loop:
        self.optimizer.zero_grad()
        self.global_step += 1
        train_input_ids = train_batch['input_ids'].to(self.device)
        train_attention_mask = train_batch['attention_mask'].to(self.device)
        train_labels = train_batch['labels'].to(self.device)
        """ forward """
        output = self.model(input_ids=train_input_ids, 
                            attention_mask=train_attention_mask, 
                            labels =train_labels)
        train_loss = output.loss
        train_total_loss += train_loss.item()
        """ record training log """
        if self.tensorboard_writer != None:
          self.tensorboard_writer.add_scalar("train/loss", train_total_loss/ (batch_id+1), self.global_step)
        """ backward """
        train_loss.backward()
        """ update """
        self.optimizer.step()
        
        """ validation loss at end of epoch"""
        if self.valid_loader != None and batch_id == len(self.train_loader) - 1:
          valid_mean_loss = self.evaluate()
          if self.tensorboard_writer != None:
            self.tensorboard_writer.add_scalar("valid/loss", valid_mean_loss, self.global_step)
        """ print """
        train_mean_loss = train_total_loss / (batch_id+1)
        loop.set_description(f'Epoch [{epoch + 1}/{self.n_epochs}]')
        loop.set_postfix(train_loss=train_mean_loss, valid_loss=valid_mean_loss)
        
      """ end of epoch """
      if self.save_model_mode == 'all':
        self.save_model(epoch, train_mean_loss, valid_mean_loss)
      elif self.save_model_mode == 'best':
        if epoch == 0 or valid_mean_loss < self.best_loss:
          self.save_model(epoch, train_mean_loss, valid_mean_loss)
          
      self.best_loss = min(self.best_loss, valid_mean_loss)
            
  def save_model(self, epoch, train_loss, valid_loss):
    torch.save(self.model.state_dict(), 
               os.path.join(self.save_model_path, 
                            f'Epoch-{epoch}_trainloss-{train_loss:.4f}_validloss-{valid_loss:.4f}.pth'))


def evaluate_entity(entity_type:str, pred:pd.DataFrame, gold:pd.DataFrame) -> Dict[str, float]:
  """ 
  This function inputs an entity type to evaluate, and predicted entities and gold standard entities
  Outputs a dictionary of exact and partial P, R, F1.
  """
  def F1(p, r):
    return 2*p*r/(p+r)

  p = pred.loc[pred['pred']==entity_type]
  g = gold.loc[gold['label']==entity_type]
  """ Exact matching """
  count = 0
  for _, p_row in p.iterrows():
    for _, g_row in g.iterrows():
      if p_row['document_id'] == g_row['document_id'] and \
        p_row['start'] == g_row['start'] and p_row['end'] == g_row['end']:
        count += 1
        break
  precision_exact = count/p.shape[0] if p.shape[0] > 0 else float('nan')
      
  count = 0
  for _, g_row in g.iterrows():
    for _, p_row in p.iterrows():
      if p_row['document_id'] == g_row['document_id'] and \
        p_row['start'] == g_row['start'] and p_row['end'] == g_row['end']:
        count += 1
        break
  recall_exact = count/g.shape[0] if g.shape[0] > 0 else float('nan')
  
  """ Partial matching """
  count = 0
  for _, p_row in p.iterrows():
    for _, g_row in g.iterrows():
      if not(p_row['end'] < g_row['start'] or p_row['start'] > g_row['end']):
        count += 1
        break
  precision_partial = count/p.shape[0] if p.shape[0] > 0 else float('nan')
    
  count = 0
  for _, g_row in g.iterrows():
    for _, p_row in p.iterrows():
      if not(p_row['end'] < g_row['start'] or p_row['start'] > g_row['end']):
        count += 1
        break
  recall_partial = count/g.shape[0] if g.shape[0] > 0 else float('nan')
    
  return {'PRECISION_EXACT':precision_exact, 
          'RECALL_EXACT':recall_exact, 
          'F1_EXACT':F1(precision_exact, recall_exact),
          'PRECISION_PARTIAL':precision_partial, 
          'RECALL_PARTIAL':recall_partial,
          'F1_PARTIAL':F1(precision_partial, recall_partial)}