from __future__ import print_function
import sklearn

import lime

import os
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.utils import resample
from scipy.special import softmax

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification

import torch
import torch.nn as nn

#%%

# Set Directory as appropiate
df_RR = pd.read_csv('Dataset/Restaurant Reviews/processed_data/Preprocess.csv')

#%%

#Loading the model from google storage and saving the models into the current directory

os.system('wget https://storage.googleapis.com/bert_model123/bert.pt')
os.system('wget https://storage.googleapis.com/bert_model123/roberta.pt')
os.system('wget https://storage.googleapis.com/bert_model123/albert.pt')

#%%
tokenizer_B = BertTokenizer.from_pretrained('bert-base-uncased',num_labels=2)
model_B = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,num_labels=2)


tokenizer_RB = RobertaTokenizer.from_pretrained('roberta-base',num_labels=2)
model_RB = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,num_labels=2)

tokenizer_AB = AlbertTokenizer.from_pretrained('albert-base-v2',num_labels=2)
model_AB = AlbertForSequenceClassification.from_pretrained('albert-base-v2', return_dict=True,num_labels=2)


#%%

#Either load the models from google storage or the one trained in Train.py

# Set Directory as appropiate
model_B.load_state_dict(torch.load("Dataset/Restaurant Reviews/model/r_bert.pt"))

model_RB.load_state_dict(torch.load("Dataset/Restaurant Reviews/model/r_roberta.pt"))

model_AB.load_state_dict(torch.load("Dataset/Restaurant Reviews/model/r_albert.pt"))



# %%
lB_loss = []
lB_logits = []
lB_label = []

for i in range(len(df_RR)):
    print(df_RR.iloc[i, 1],df_RR.iloc[i,0])

    inputs_B = tokenizer_B(df_RR.iloc[i, 0], return_tensors="pt")

    labels = torch.tensor([df_RR.iloc[i,1]]).unsqueeze(0)   # Batch size 1

    outputs_B = model_B(**inputs_B, labels=labels)

    B_loss = outputs_B.loss
    B_logits = outputs_B.logits
    var = int(B_logits.argmax().detach())

    lB_label.append(var)
    lB_loss.append(B_loss)
    lB_logits.append(B_logits)

df_RR
#%%
df_RR['Bert_Loss'] = lB_loss
df_RR['Bert_Logits'] = lB_logits
df_RR['Bert_Labels'] = lB_label

df_RR
#%%
df_RR.to_csv('Dataset/Restaurant Reviews/processed_data/predict.csv', index=False)

#%%

df_RR = pd.read_csv('Dataset/Restaurant Reviews/processed_data/predict.csv')

# %%
lRB_loss = []
lRB_logits = []
lRB_label = []

for i in range(len(df_RR)):
    print(df_RR.iloc[i, 1],df_RR.iloc[i,0])

    inputs_RB = tokenizer_RB(df_RR.iloc[i, 0], return_tensors="pt")

    labels = torch.tensor([df_RR.iloc[i,1]]).unsqueeze(0)   # Batch size 1

    outputs_RB = model_RB(**inputs_RB, labels=labels)

    RB_loss = outputs_RB.loss
    RB_logits = outputs_RB.logits
    var = int(RB_logits.argmax().detach())

    lRB_label.append(var)
    lRB_loss.append(RB_loss)
    lRB_logits.append(RB_logits)

df_RR
#%%
df_RR['roberta_Loss'] = lRB_loss
df_RR['roberta_Logits'] = lRB_logits
df_RR['roberta_Labels'] = lRB_label

df_RR
#%%
df_RR.to_csv('Dataset/Restaurant Reviews/processed_data/predict.csv', index=False)

#%%

df_RR = pd.read_csv('Dataset/Restaurant Reviews/processed_data/predict.csv')

# %%
lAB_loss = []
lAB_logits = []
lAB_label = []

for i in range(len(df_RR)):
    print(df_RR.iloc[i, 1],df_RR.iloc[i,0])

    inputs_AB = tokenizer_AB(df_RR.iloc[i, 0], return_tensors="pt")

    labels = torch.tensor([df_RR.iloc[i,1]]).unsqueeze(0)   # Batch size 1

    outputs_AB = model_AB(**inputs_AB, labels=labels)

    AB_loss = outputs_AB.loss
    AB_logits = outputs_AB.logits
    var = int(AB_logits.argmax().detach())

    lAB_label.append(var)
    lAB_loss.append(AB_loss)
    lAB_logits.append(AB_logits)

df_RR
#%%
df_RR['albert_Loss'] = lAB_loss
df_RR['albert_Logits'] = lAB_logits
df_RR['albert_Labels'] = lAB_label

df_RR
#%%

df_RR.to_csv('Dataset/Amazon Food Reviews/processed_data/predict.csv', index=False)