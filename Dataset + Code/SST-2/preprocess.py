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


# %%

# Set Directory as appropiate
df_ST = pd.read_csv("original_data/train.tsv", sep="\t",nrows=1500)


sns.countplot(df_ST['label'])

plt.show()

#%%

zero = df_ST[df_ST.label==0]
one = df_ST[df_ST.label==1]


# downsample majority

one_u = resample(one,
                  replace=True, # sample with replacement
                  n_samples=len(zero), # match number with majority class
                  random_state=42) # reproducible results


# combine majority and upsampled minority
df_ST = pd.concat([zero,one_u])
#%%
sns.countplot(df_ST['label'])
plt.show()
#%%

df_ST.to_csv('processed_data/Preprocess.csv', index=False)