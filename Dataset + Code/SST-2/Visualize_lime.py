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
from scipy.special import softmax
import pickle

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification

import torch
import torch.nn as nn

#%%
import lime

from lime import lime_text
from lime.lime_text import LimeTextExplainer

#%%

tokenizer_B = BertTokenizer.from_pretrained('bert-base-uncased',num_labels=2)
model_B = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,num_labels=2)

#%%

#Loading the model from google storage and saving the models into the current directory


#%%

#Either load the models from google storage or the one trained in Train.py

model_B.load_state_dict(torch.load("Dataset/SST-2/model/bert.pt"))



#%%

df_SR = pd.read_csv('Dataset/SST-2/processed_data/Preprocess.csv')
df_SR

# %%

#Change to appropiate model in the class

class Prediction_Transformer:

    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predictor(self, texts):
        results = []
        for text in texts:
            # labels = torch.tensor([1]).unsqueeze(0)
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits

            res = softmax(logits.detach().numpy()[0])

            results.append(res)

        ress = [res for res in results]
        results_array = np.array(ress)
        return results_array

# %%
explainer = LimeTextExplainer(class_names=[0, 1])

prediction_B = Prediction_Transformer(model_B,tokenizer_B)



#%%

c = 150

for i in range(0,1):
    #BERT
    text = df_SR.iloc[c, 0]  # Example text
    exp = explainer.explain_instance(text, prediction_B.predictor, labels=(0, 1), num_features=5,
                                     num_samples=len(text.split()))
    exp.show_in_notebook(text=True)

    exp.save_to_file('Dataset/SST-2/results/bert_example{}.html'.format(i))

