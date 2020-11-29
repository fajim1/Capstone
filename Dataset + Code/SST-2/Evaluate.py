from __future__ import print_function
import sklearn

#%%
import lime
#%%
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
#%%
import os
os.chdir('/home/ubuntu/transformers/Dataset/SST-2')
#%%

# Set Directory as appropiate
df_ST = pd.read_csv('processed_data/predict.csv')
df_ST
#%%

y_test = df_ST['label']
y_pred = df_ST['Bert_Labels']

print("BERT")
#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))


from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2']))
print('\n')

#%%
