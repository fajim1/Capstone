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
df_MR = pd.read_csv('Movie Reviews/train.tsv/train.tsv',delimiter='\t',nrows=1000)

df_MR

#%%
df_MR = df_MR.iloc[:,[]]


#%%

zero = df_AR[df_AR.Score==0]
one = df_AR[df_AR.Score==1]
two = df_AR[df_AR.Score==2]
three = df_AR[df_AR.Score==3]
four = df_AR[df_AR.Score==4]

# upsample minority
four_u = resample(four,
                  replace=True, # sample with replacement
                  n_samples=len(three), # match number with majority class
                  random_state=42) # reproducible results


# combine majority and upsampled minority
df_AR = pd.concat([zero,one,two,three,four_u])
#%%
sns.countplot(df_AR['Score'])
plt.show()
#%%

df_AR.to_csv('Dataset/Amazon Food Reviews/processed_data/Preprocess.csv', index=False)