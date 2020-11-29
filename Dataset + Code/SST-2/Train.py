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
#%%
import os
os.chdir('/home/ubuntu/transformers/Dataset/SST-2')

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %%

# Set Directory as appropiate
df_ST = pd.read_csv('processed_data/Preprocess.csv')


sns.countplot(df_ST['label'])

plt.show()

#%%
txt = []

for i in np.array(df_ST.iloc[:,0]):
    txt.append(i)

#%%


tokenizer_B = BertTokenizer.from_pretrained('bert-base-uncased',num_labels=2)
model_B = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,num_labels=2)
model_B.to(device)


#%%
model_B.train()


#%%
from transformers import AdamW
optimizer = AdamW(model_B.parameters(), lr=1e-4)

#%%

for param in model_B.base_model.parameters():
    param.requires_grad = False


#%%
text_batch = txt
label_batch = np.array(df_ST.iloc[:,1])
#%%
for epochs in range(30):
    loss_a = []
    for text,label in zip(text_batch,label_batch):
        encoding = tokenizer_B(text, return_tensors='pt', padding=True, truncation=True)

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels = torch.tensor([label]).unsqueeze(0).to(device)

        outputs = model_B(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss_a.append(loss.detach().cpu())


        loss.backward()
        optimizer.step()

    if epochs%1==0:
        loss_a = np.array(loss_a)
        print("After {} epochs, loss is {}".format(epochs,np.mean(loss_a)))


#%%

# Set Directory as appropiate

torch.save(model_B.state_dict(), "model/bert.pt")


#%%

