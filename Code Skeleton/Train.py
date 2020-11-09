from __future__ import print_function
import sklearn

#%%
import lime
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)
import sklearn
import sklearn.ensemble
import sklearn.metrics
import seaborn as sns
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.utils import resample

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification

import torch
import torch.nn as nn

import torch


# %%

df_AR = pd.read_csv('Dataset/Amazon Food Reviews/processed_data/Preprocess.csv')


sns.countplot(df_AR['Score'])
plt.show()

#%%
txt = []

for i in np.array(df_AR.iloc[:,1]):
    txt.append(i)

#%%

# tokenizer_B = BertTokenizer.from_pretrained('bert-base-uncased',num_labels=5)
# model_B = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,num_labels=5)


tokenizer_RB = RobertaTokenizer.from_pretrained('roberta-base',num_labels=5)
model_RB = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,num_labels=5)

# tokenizer_AB = AlbertTokenizer.from_pretrained('albert-base-v2',num_labels=5)
# model_AB = AlbertForSequenceClassification.from_pretrained('albert-base-v2', return_dict=True,num_labels=5)

#%%
# model_B.train()

model_RB.train()

# model_AB.train()

#%%
from transformers import AdamW
optimizer = AdamW(model_RB.parameters(), lr=1e-4)

#%%

for param in model_RB.base_model.parameters():
    param.requires_grad = False


#%%
text_batch = txt
label_batch = np.array(df_AR.iloc[:,0])
#%%
for epochs in range(20):
    loss_a = []
    for text,label in zip(text_batch,label_batch):
        encoding = tokenizer_RB(text, return_tensors='pt', padding=True, truncation=True)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.tensor([label]).unsqueeze(0)

        outputs = model_RB(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss_a.append(loss.detach())


        loss.backward()
        optimizer.step()

    if epochs%1==0:
        loss_a = np.array(loss_a)
        print("After {} epochs, loss is {}".format(epochs,np.mean(loss_a)))

#%%
#
# torch.save(model_B.state_dict(), "Dataset/Amazon Food Reviews/model/bert.pt")
torch.save(model_RB.state_dict(), "Dataset/Amazon Food Reviews/model/roberta.pt")
# torch.save(model_AB.state_dict(), "Dataset/Amazon Food Reviews/model/albert.pt")
#%%
#
#
# %%
