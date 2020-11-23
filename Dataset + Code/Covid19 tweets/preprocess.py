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

# Set Directory as appropiate
df_CR = pd.read_csv('Dataset/Covid19 tweets/Corona_NLP_train.csv',nrows = 1000)

df_CR = df_CR.iloc[:,[4,5]]
df_CR

#%%
sns.countplot(df_CR['Sentiment'])
plt.show()
#%%
for i in range(len(df_CR.iloc[:,1])):

    if df_CR.iloc[i,1] == 'Extremely Negative':
        df_CR.iloc[i, 1] = 0

    if df_CR.iloc[i,1] == 'Negative':
        df_CR.iloc[i, 1] = 1

    if df_CR.iloc[i,1] == 'Neutral':
        df_CR.iloc[i, 1] = 2

    if df_CR.iloc[i,1] == 'Positive':
        df_CR.iloc[i, 1] = 3

    if df_CR.iloc[i,1] == 'Extremely Positive':
        df_CR.iloc[i, 1] = 4


df_CR
# from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# le.fit(['Extremely Negative', 'Negative','Neutral', 'Positive','Extremely Positive'])
# df_CR["Sentiment"]=le.transform(df_CR["Sentiment"])
#
# df_CR
#%%

zero = df_CR[df_CR.Sentiment==0]
one = df_CR[df_CR.Sentiment==1]
two = df_CR[df_CR.Sentiment==2]
three = df_CR[df_CR.Sentiment==3]
four = df_CR[df_CR.Sentiment==4]

# Downsample Majority
one_u = resample(one,
                  replace=True, # sample with replacement
                  n_samples=len(zero), # match number with majority class
                  random_state=42) # reproducible results

three_u = resample(three,
                  replace=True, # sample with replacement
                  n_samples=len(zero), # match number with majority class
                  random_state=42) # reproducible results

# combine majority and upsampled minority
df_CR = pd.concat([zero,one_u,two,three_u,four])
#%%
sns.countplot(df_CR['Sentiment'])
plt.show()
#%%

df_CR.to_csv('Dataset/Covid19 tweets/processed_data/Preprocess.csv', index=False)