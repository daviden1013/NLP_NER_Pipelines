# -*- coding: utf-8 -*-
from typing import List, Dict
import string
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification
from modules.Utilities import word_tokenize, Tokens_to_entities
from tqdm import tqdm


class Data_holder:
  def __init__(self, doc_dict: Dict[str, str], label_map:Dict[str, int]):
    """ 
    This class holds the input document as dict {document_id:content}
    Creates word tokens (BIO) and feed NER predictor
    It also handles outputs and make predicted tokens into entity
    """
    self.doc_dict = doc_dict
    self.label_map = label_map
    """ Create BIO dict {document_id:list of word tokens} """
    self.bio = {}
    for doc_id, txt in self.doc_dict.items():
      self.bio[doc_id] = word_tokenize(txt)
  
  def _get_entity_text(self, document_id:str, start:int, end:int) -> str:
    return self.doc_dict[document_id][start:end]
  
  def Predict_to_entity(self, token_pred_df: pd.DataFrame, mode:str) -> pd.DataFrame:
    """
    This method inputs a df of token predictions from predictor object
    outputs a df of ['document_id', 'entity', 'start', 'end', 'pred', 'prob', 'conf']

    Parameters
    ----------
    token_pred_df : pd.DataFrame
      df of token predictions from predictor object
    mode : str
      one of {'BIO', 'IO'}

    Returns
    -------
    out : TYPE
      DESCRIPTION.

    """
    """ process prediction """
    assert mode in {'BIO', 'IO'}, "mode must be one of {'BIO', 'IO'}"
    
    token_pred_df['pred'] = token_pred_df[[f'prob_{v}' for v in self.label_map.keys()]].\
      idxmax(axis=1).str.replace('prob_', '')
    token_pred_df['prob'] = token_pred_df[[v for v in token_pred_df.columns if v[:5] == 'prob_']].max(axis=1)
      
    token_pred_df = token_pred_df.sort_values(['document_id', 'start', 'end', 'prob']). \
      groupby(['document_id', 'start', 'end']).last()
      
    token_pred_df = token_pred_df[['pred', 'prob']]
    token_pred_df.reset_index(inplace=True)
    
    entity = Tokens_to_entities(token_df=token_pred_df, mode=mode, 
                              label_varname='pred', n_level=len(self.label_map))
    
    entity['entity'] = entity.apply(lambda df:self._get_entity_text(df.document_id, df.start, df.end), axis=1)
    return entity
  

class NER_Dataset(Dataset):
  def __init__(self, 
               bios: Dict, 
               tokenizer: BertTokenizer, 
               label_map: Dict,
               word_seq_lenght: int=32, 
               step: int=10,
               token_seq_length: int=64):
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
    """
    self.tokenizer = tokenizer
    self.label_map = label_map
    self.word_seq_lenght = word_seq_lenght
    self.step = step
    self.token_seq_length = token_seq_length  
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
    out = {'input_ids':[],
           'attention_mask':[],
           'start':[],
           'end':[]}
    
    for word in word_seq:
      tokens = self.tokenizer.tokenize(word[0])
      if len(tokens) == 0:
        continue
      input_ids = self.tokenizer.encode(tokens, add_special_tokens=False)
      
      out['input_ids'].extend(input_ids)
      out['attention_mask'].extend([1]*len(tokens))
      out['start'].extend([word[1]]*len(tokens))
      out['end'].extend([word[2]]*len(tokens))
      
    # truncate or padding to make token lenght = self.token_seq_length
    if len(out['input_ids']) > self.token_seq_length:
      for k in out.keys():
        out[k] = out[k][0:self.token_seq_length]
      
    else:
      for _ in range(self.token_seq_length - len(out['input_ids'])):
        out['input_ids'].append(0)
        out['attention_mask'].append(0)
        out['start'].append(-1)
        out['end'].append(-1)
            
    out['document_id'] = [self.document_ids[idx]] * len(out['input_ids'])
    
    # make input_ids, attention_mask, (labels), start, end tensor
    for v in out.keys():
      if v != 'document_id':
        out[v] = torch.tensor(out[v])
      
    return out


class NER_Predictor:
  def __init__(self, 
               model:AutoModelForTokenClassification,
               tokenizer:BertTokenizer, 
               dataset: Dataset,
               label_map:Dict,
               batch_size:int):
    
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.model = model
    self.model.to(self.device)
    self.model.eval()
    self.tokenizer = tokenizer
    self.label_map = label_map
    self.batch_size = batch_size
    self.dataset = dataset
    self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

  def predict(self) -> pd.DataFrame:
    token_pred = {'document_id':[],
                 'input_ids':[],
                 'start':[],
                 'end':[]}
    for tag in self.label_map.keys():
      token_pred[tag] = []
    
    loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=True)
    for i, ins in loop:  
      input_ids = ins['input_ids'].to(self.device)
      attention_mask = ins['attention_mask'].to(self.device)
      with torch.no_grad():
        p = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pred = p.logits
        for b in range(pred.shape[0]):
          # token info
          token_pred['document_id'].extend([i[b] for i in ins['document_id']])
          token_pred['input_ids'].extend(ins['input_ids'][b].cpu().tolist())
          token_pred['start'].extend(ins['start'][b].cpu().tolist())
          token_pred['end'].extend(ins['end'][b].cpu().tolist())
            
          # predicted logits for each level
          for tag, code in self.label_map.items():
            token_pred[tag].extend(pred[b,:,code].cpu().tolist())
            
    token_pred_df = pd.DataFrame(token_pred)
    token_pred_df = token_pred_df.loc[token_pred_df['start'] != -1]

    # convert logit to probability
    minimum_logit = token_pred_df[self.label_map.keys()].min(axis=1)
    for tag in self.label_map.keys():
      token_pred_df[tag] = token_pred_df[tag] - minimum_logit
      
    sum_logit = token_pred_df[self.label_map.keys()].sum(axis=1)
    for tag in self.label_map.keys():
      token_pred_df[tag] = token_pred_df[tag]/sum_logit
      
    token_pred_df.rename(columns={tag:f'prob_{tag}' for tag in self.label_map.keys()}, inplace=True)
    return token_pred_df
  
  