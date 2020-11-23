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

tokenizer_B = BertTokenizer.from_pretrained('bert-base-uncased',num_labels=5)
model_B = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True,num_labels=5)


tokenizer_RB = RobertaTokenizer.from_pretrained('roberta-base',num_labels=5)
model_RB = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,num_labels=5)

tokenizer_AB = AlbertTokenizer.from_pretrained('albert-base-v2',num_labels=5)
model_AB = AlbertForSequenceClassification.from_pretrained('albert-base-v2', return_dict=True,num_labels=5)

#%%

#Loading the model from google storage and saving the models into the current directory

os.system('wget https://storage.googleapis.com/bert_model123/bert.pt')
os.system('wget https://storage.googleapis.com/bert_model123/roberta.pt')
os.system('wget https://storage.googleapis.com/bert_model123/albert.pt')

#%%

#Either load the models from google storage or the one trained in Train.py

model_B.load_state_dict(torch.load("Dataset/Amazon Food Reviews/bert.pt"))

model_RB.load_state_dict(torch.load("Dataset/Amazon Food Reviews/roberta.pt"))

model_AB.load_state_dict(torch.load("Dataset/Amazon Food Reviews/albert.pt"))

#%%

df_AR = pd.read_csv('Dataset/Amazon Food Reviews/processed_data/predict.csv')


# %%

#Change to appropiate model in the class

class Prediction:

    def __init__(self):
        self.model = model_B
        self.tokenizer = tokenizer_B

    def predictor(self, texts):
        results = []
        for text in texts:
            # labels = torch.tensor([1]).unsqueeze(0)
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits

            res = softmax(logits.detach().numpy())[0]

            results.append(res)

        ress = [res for res in results]
        results_array = np.array(ress)
        return results_array


# %%
explainer = LimeTextExplainer(class_names=[0, 1, 2, 3, 4])

prediction = Prediction()
#%%
text = df_AR.iloc[925, 1] # Example text
exp = explainer.explain_instance(text, prediction.predictor, labels=(0, 1, 2, 3, 4), num_features=5,
                                 num_samples=len(text.split()))
exp.show_in_notebook(text=True)

# %%

# Set Directory as appropiate
exp.save_to_file('Dataset/Amazon Food Reviews/html/label3.html')

