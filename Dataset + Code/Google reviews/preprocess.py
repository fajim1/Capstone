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
df_GR = pd.read_csv('Dataset/Google reviews/original_data/reviews.csv',nrows = 1200)

df_GR = df_GR.iloc[:,[2,3]]

df_GR['score'] = df_GR['score']-1
#%%
sns.countplot(df_GR['score'])

plt.show()

#%%

zero = df_GR[df_GR.score==0]
one = df_GR[df_GR.score==1]
two = df_GR[df_GR.score==2]
three = df_GR[df_GR.score==3]
four = df_GR[df_GR.score==4]

# downsample majority
zero_u = resample(zero,
                  replace=True, # sample with replacement
                  n_samples=150, # match number with majority class
                  random_state=42) # reproducible results

one_u = resample(one,
                  replace=True, # sample with replacement
                  n_samples=150, # match number with majority class
                  random_state=42) # reproducible results

two_u = resample(two,
                  replace=True, # sample with replacement
                  n_samples=150, # match number with majority class
                  random_state=42) # reproducible results

three_u = resample(three,
                  replace=True, # sample with replacement
                  n_samples=150, # match number with majority class
                  random_state=42) # reproducible results

four_u = resample(four,
                  replace=True, # sample with replacement
                  n_samples=150, # match number with majority class
                  random_state=42) # reproducible results


# combine majority and upsampled minority
df_GR = pd.concat([zero_u,one_u,two_u,three_u,four_u])
#%%
sns.countplot(df_GR['score'])

plt.show()
#%%

df_GR.to_csv('Dataset/Google reviews/processed_data/Preprocess.csv', index=False)