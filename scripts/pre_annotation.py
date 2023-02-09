# -*- coding: utf-8 -*-
import abc
import os
from typing import List, Dict
import pandas as pd
import json

class Label_studio_preannotator:
  def __init__(self, docs:pd.DataFrame, DOC_ID:str):
    """
    This class inputs a DataFrame of documents (should have document_id, text, 
    and any columns that want to import to Label studio). Outputs a JSON with 
    preannotations that meets Label studio's import format. 

    Parameters
    ----------
    docs : pd.DataFrame
      DataFrame with documents
    DOC_ID : str
      column name for document id
    """
    assert 'text' in docs.columns, 'A "text" column with document content is required.'
    self.docs = docs
    self.DOC_ID = DOC_ID
    self.ann = {}
    
  @abc.abstractmethod  
  def pre_annotate(self):
    """
    This method annotate the documents with algorithm (e.g., keyword matching)
    Outputs a dict of annotation self.ann
      {document_id:[{start, end, text, [labels]}]}
    """
    return NotImplemented
  
    
  def get_JSON(self, cols:List[str]=None) -> str:
    """ 
    This method 
    """
    if cols == None:
      cols = self.docs.columns
    elif 'text' not in cols:
      cols.append('text')
    
    out = []
    for r in self.docs.itertuples():
      data = {c:getattr(r, c) for c in cols}
      ann = self.ann[getattr(r, self.DOC_ID)]
      
      results = []
      for i, a in enumerate(ann): 
        res = {"id": f"{getattr(r, self.DOC_ID)}_{i}",
               "from_name": "label",
               "to_name": "text",
               "type": "labels",
               "value": {
                  "start": a['start'],
                  "end": a['end'],
                  "text": a['text'],
                  "labels": a['labels']}
              }
        results.append(res)
      
      pred = {"model_version": "one",
              "result": results}
      
      out.append({'data':data, 'predictions':[pred]})
     
    return json.dumps(out)

    
class Medication_NER_preannotator(Label_studio_preannotator):
  def __init__(self, docs:pd.DataFrame, DOC_ID:str):
    super().__init__(docs, DOC_ID)
    
  def pre_annotate(self, keywords:List[str]):
    pattern = re.compile('|'.join(keywords))
    for r in self.docs.itertuples():
      self.ann[getattr(r, self.DOC_ID)] = [{'start':it.start(), 'end':it.end(), 'text':it.group(), 'labels':['HIGHLIGHT']} \
                                             for it in re.finditer(pattern, r.text)]
        
        
""" Pipeline """
files = os.listdir(os.path.join(PATH, 'data', 'text'))
txt_list = []
for file in files:
  with open(os.path.join(PATH, 'data', 'text', file), 'r', encoding='utf-8') as f:
    txt = f.read()

  txt_list.append((file.replace('.txt', ''), txt))
  
txt_df = pd.DataFrame(txt_list, columns=['document_id', 'text'])

prea = Medication_NER_preannotator(docs=txt_df.head(2), DOC_ID='document_id')
prea.pre_annotate(['IV', 'IVF', 'IV Fluids', 'PRBCs', 'PRN', 'QD', 'bid', 'PO', 'Gtt']) 
json_file = prea.get_JSON(cols=['document_id', 'text'])
        
with open(os.path.join(PATH, 'Label_studio_input.json'), 'w') as f:
  f.write(json_file)


