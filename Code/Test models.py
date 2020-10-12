from __future__ import print_function
import sklearn
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
import sklearn
import sklearn.ensemble
import sklearn.metrics

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import torch

tokenizer_B = BertTokenizer.from_pretrained('bert-base-uncased')
model_B = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)

tokenizer_RB = RobertaTokenizer.from_pretrained('roberta-base')
model_RB = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)

# %%
df_MR = pd.read_csv('Dataset/Movie Reviews/train.tsv/train.tsv',delimiter='\t',nrows=1000)
df_MR

# %%
lB_loss = []
lB_logits = []

lRB_loss = []
lRB_logits = []

for i in range(len(df_MR)):
    print(df_MR.iloc[i, 2])
    inputs_B = tokenizer_B(df_MR.iloc[i, 2], return_tensors="pt")
    inputs_RB = tokenizer_RB(df_MR.iloc[i, 2], return_tensors="pt")

    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

    outputs_B = model_B(**inputs_B, labels=labels)
    outputs_RB = model_RB(**inputs_RB, labels=labels)

    B_loss = outputs_B.loss
    B_logits = outputs_B.logits
    lB_loss.append(B_loss)
    lB_logits.append(B_logits)


    RB_loss = outputs_RB.loss
    RB_logits = outputs_RB.logits
    lRB_loss.append(RB_loss)
    lRB_logits.append(RB_logits)

RB_logits
#%%
df_MR['Bert_Loss'] = lB_loss
df_MR['Bert_Logits'] = lB_logits

df_MR['RoBerta_Loss'] = lRB_loss
df_MR['RoBerta_Logits'] = lRB_logits

df_MR
#%%

df_MR.to_csv('Dataset/Movie Reviews/results.csv', index=False)

#%%

df_RR = pd.read_csv('Dataset/Restaurant Reviews/Restaurant_Reviews.tsv',delimiter='\t',header=0,nrows=500)
df_RR

#%%
lB_loss = []
lB_logits = []

lRB_loss = []
lRB_logits = []
for i in range(len(df_RR)):
  print(df_RR.iloc[i,0])
  inputs_B = tokenizer_B(df_RR.iloc[i,0], return_tensors="pt")
  inputs_RB = tokenizer_RB(df_RR.iloc[i,0], return_tensors="pt")

  labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

  outputs_B = model_B(**inputs_B, labels=labels)
  outputs_RB = model_RB(**inputs_RB, labels=labels)

  B_loss = outputs_B.loss
  B_logits = outputs_B.logits
  lB_loss.append(B_loss)
  lB_logits.append(B_logits)

  RB_loss = outputs_RB.loss
  RB_logits = outputs_RB.logits
  lRB_loss.append(RB_loss)
  lRB_logits.append(RB_logits)

lRB_logits
#%%
df_RR['Bert_Loss'] = lB_loss
df_RR['Bert_Logits'] = lB_logits

df_RR['RoBerta_Loss'] = lRB_loss
df_RR['RoBerta_Logits'] = lRB_logits

df_RR

#%%

df_RR.to_csv('Dataset/Restaurant Reviews/results.csv', index=False)

#%%
df_AR = pd.read_csv('Dataset/Amazon Food Reviews/Reviews.csv',nrows=500)
df_AR
#%%
lB_loss = []
lB_logits = []

lRB_loss = []
lRB_logits = []
for i in range(len(df_AR)):
  print(df_AR.iloc[i,8])
  inputs_B = tokenizer_B(df_AR.iloc[i,8], return_tensors="pt")
  inputs_RB = tokenizer_RB(df_AR.iloc[i,8], return_tensors="pt")

  labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

  outputs_B = model_B(**inputs_B, labels=labels)
  outputs_RB = model_RB(**inputs_RB, labels=labels)

  B_loss = outputs_B.loss
  B_logits = outputs_B.logits
  lB_loss.append(B_loss)
  lB_logits.append(B_logits)

  RB_loss = outputs_RB.loss
  RB_logits = outputs_RB.logits
  lRB_loss.append(RB_loss)
  lRB_logits.append(RB_logits)


#%%
df_AR['Bert_Loss'] = lB_loss
df_AR['Bert_Logits'] = lB_logits

df_AR['RoBerta_Loss'] = lRB_loss
df_AR['RoBerta_Logits'] = lRB_logits

df_AR
#%%
df_AR.to_csv('results.csv', index=False)


#%%