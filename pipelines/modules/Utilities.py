# -*- coding: utf-8 -*-
from typing import List, Dict
import pandas as pd
import string

def word_tokenize(text:str) -> List:
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
  i = 0
  out = []
  for j, c in enumerate(text):  
    if c in ' \n' + string.punctuation:
      if i < j and text[i] not in [' ', '\n']:
        out.append((text[i:j], i, j))
      i = j
      j += 1
      if i < j and text[i] not in [' ', '\n']:
        out.append((text[i:j], i, j))
      i = j
      
    j += 1
  return out


def Tokens_to_entities(token_df: pd.DataFrame, mode:str, label_varname:str, n_level:int) -> pd.DataFrame:
  """
  This method inputs a df of tokens
  outputs a df of ['document_id', 'entity', 'start', 'end', label_varname, 'prob', 'conf']

  Parameters
  ----------
  token_df : pd.DataFrame
    df of tokens. Must include columns: document_id, start, end, and [label_varname]. 
    Optional columns: prob
  mode : str
    one of {'BIO', 'IO'}
  label_varname : str
    Variable name of predicted label or annotated label.

  Returns
  -------
  out :  pd.DataFrame
   Entity list. df of ['document_id', 'entity', 'start', 'end', label_varname, ('prob', 'conf')]

  """
  assert mode in {'BIO', 'IO'}, "mode must be one of {'BIO', 'IO'}"
  if mode == 'BIO':
    entity_chunk = (token_df[label_varname].str.match('^B-|^O')).cumsum()
  else:
    entity_chunk = (token_df[label_varname] != token_df[label_varname].shift()).cumsum()
    
  agg_vars = {'start':'min', 'end':'max', label_varname:'max'}
  if 'prob' in token_df.columns:
    agg_vars['prob'] = 'mean'
    entity = token_df.groupby(['document_id', entity_chunk], sort=False, as_index=False). \
      agg(agg_vars)
  
    entity = entity.loc[entity[label_varname] != 'O']
    entity[label_varname] = entity[label_varname].str.replace('B-', '').str.replace('I-', '')
    # confident = prob / baseline prob
    entity['conf'] = entity['prob'] * n_level
    return entity.reindex(['document_id', 'start', 'end', label_varname, 'prob', 'conf'], axis=1)
    
  else:
    entity = token_df.groupby(['document_id', entity_chunk], sort=False, as_index=False). \
      agg(agg_vars)
  
    entity = entity.loc[entity[label_varname] != 'O']
    entity[label_varname] = entity[label_varname].str.replace('B-', '').str.replace('I-', '')
    return entity.reindex(['document_id', 'start', 'end', label_varname], axis=1)