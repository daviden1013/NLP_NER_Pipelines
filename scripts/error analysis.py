# -*- coding: utf-8 -*-

import pandas as pd
import openpyxl

doc = pd.read_csv(os.path.join(PATH, '20230222 sample.csv'))
doc_dict = {i:t for i, t in zip(doc['Order ID'], doc['text'])}

gold = pd.read_pickle(os.path.join(PATH, 'evaluations', 'run2', 'run2 entity_gold.pickle'))
pred = pd.read_pickle(os.path.join(PATH, 'evaluations', 'run2', 'run2 entity_pred.pickle'))

gold.sort_values(['document_id', 'start'], inplace=True)
pred.sort_values(['document_id', 'start'], inplace=True)

def get_entity_text(document_id:str, start:int, end:int) -> str:
  return doc_dict[document_id][start:end]

gold['entity_gold'] = gold.apply(lambda x:get_entity_text(x.document_id, x.start, x.end), axis=1)
pred['entity_pred'] = pred.apply(lambda x:get_entity_text(x.document_id, x.start, x.end), axis=1)

gold.rename(columns={'start':'start_gold', 'end':'end_gold'}, inplace=True)
pred.rename(columns={'start':'start_pred', 'end':'end_pred'}, inplace=True)

df = pd.merge(pred, gold, left_on=['document_id', 'pred'], right_on=['document_id', 'label'])
df = df.loc[~((df['end_pred'] < df['start_gold']) | (df['start_pred'] > df['end_gold']))]
exact = df.loc[(df['start_gold']==df['start_pred']) & (df['end_gold']==df['end_pred'])]
partial = df.loc[(df['start_gold']!=df['start_pred']) | (df['end_gold']!=df['end_pred'])]


type1 = pd.merge(pred, df, on=['document_id', 'start_pred', 'end_pred'], how='left', indicator=True)
type1 = type1.loc[type1['_merge']=='left_only']
type1.rename(columns={'pred_x':'pred', 'entity_pred_x':'entity_pred'}, inplace=True)
type1 = type1[['document_id', 'start_pred', 'end_pred', 'pred', 'entity_pred']]


type2 = pd.merge(gold, df, on=['document_id', 'start_gold', 'end_gold'], how='left', indicator=True)
type2 = type2.loc[type2['_merge']=='left_only']
type2.rename(columns={'label_x':'label', 'entity_gold_x':'entity_gold'}, inplace=True)
type2 = type2[['document_id', 'start_gold', 'end_gold', 'label', 'entity_gold']]

with pd.ExcelWriter(os.path.join(PATH, '20230228 Error analysis.xlsx')) as writer:
  exact.to_excel(writer, sheet_name='Exact', index=False)
  partial.to_excel(writer, sheet_name='Partial', index=False)
  type1.to_excel(writer, sheet_name='Type1', index=False)
  type2.to_excel(writer, sheet_name='Type2', index=False)
