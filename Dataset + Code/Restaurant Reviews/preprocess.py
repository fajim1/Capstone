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
df_RR = pd.read_csv('Dataset/Restaurant Reviews/original_data/Restaurant_Reviews.tsv',delimiter='\t',nrows = 10000)

sns.countplot(df_RR['Liked'])
plt.show()

#%%

zero = df_RR[df_RR.Liked==0]
one = df_RR[df_RR.Liked==1]


# upsample minority
one = resample(one,
                  replace=True, # sample with replacement
                  n_samples=len(zero), # match number with majority class
                  random_state=42) # reproducible results


# combine majority and upsampled minority
df_RR = pd.concat([zero,one])
#%%
sns.countplot(df_RR['Liked'])
plt.show()
#%%

df_RR.to_csv('Dataset/Restaurant Reviews/processed_data/Preprocess.csv', index=False)