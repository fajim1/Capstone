from __future__ import print_function
import sklearn
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import sklearn
import sklearn.ensemble
import sklearn.metrics

from transformers import BertTokenizer, BertForSequenceClassification

import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)



# %%
df_MR = pd.read_csv('train.tsv',delimiter='\t',nrows=1000)
df_MR

# %%
