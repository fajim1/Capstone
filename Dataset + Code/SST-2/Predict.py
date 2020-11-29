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
import os
os.chdir('/home/ubuntu/transformers/Dataset/SST-2')
#%%


# Set Directory as appropiate
df_ST = pd.read_csv('processed_data/Preprocess.csv')

#%%


#Loading the model from google storage and saving the models into the current directory

os.system('wget https://storage.googleapis.com/bert_model123/bert.pt')
os.system('wget https://storage.googleapis.com/bert_model123/roberta.pt')
os.system('wget https://storage.googleapis.com/bert_model123/albert.pt')

#%%
tokenizer_B = BertTokenizer.from_pretrained('bert-base-uncased',num_labels=2)
model_B = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,num_labels=2)

#%%

#Either load the models from google storage or the one trained in Train.py

# Set Directory as appropiate
model_B.load_state_dict(torch.load("model/bert.pt"))



# %%
lB_loss = []
lB_logits = []
lB_label = []

for i in range(len(df_ST)):
    print(df_ST.iloc[i, 1],df_ST.iloc[i,0])

    inputs_B = tokenizer_B(df_ST.iloc[i, 0], return_tensors="pt")

    labels = torch.tensor([df_ST.iloc[i,1]]).unsqueeze(0)   # Batch size 1

    outputs_B = model_B(**inputs_B, labels=labels)

    B_loss = outputs_B.loss
    B_logits = outputs_B.logits
    var = int(B_logits.argmax().detach())

    lB_label.append(var)
    lB_loss.append(B_loss)
    lB_logits.append(B_logits)

df_ST
#%%
df_ST['Bert_Loss'] = lB_loss
df_ST['Bert_Logits'] = lB_logits
df_ST['Bert_Labels'] = lB_label

df_ST
#%%
df_ST.to_csv('processed_data/predict.csv', index=False)

#%%

