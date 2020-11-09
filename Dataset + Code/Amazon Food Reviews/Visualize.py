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

#
# tokenizer_RB = RobertaTokenizer.from_pretrained('roberta-base',num_labels=5)
# model_RB = RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True,num_labels=5)
#
# tokenizer_AB = AlbertTokenizer.from_pretrained('albert-base-v2',num_labels=5)
# model_AB = AlbertForSequenceClassification.from_pretrained('albert-base-v2', return_dict=True,num_labels=5)



#%%
model_B.load_state_dict(torch.load("Dataset/Amazon Food Reviews/bert.pt"))

# model_RB.load_state_dict(torch.load("Dataset/Amazon Food Reviews/roberta.pt"))
#
# model_AB.load_state_dict(torch.load("Dataset/Amazon Food Reviews/albert.pt"))

#%%

df_AR = pd.read_csv('Dataset/Amazon Food Reviews/processed_data/predict.csv')


# %%
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
text = df_AR.iloc[925, 1]
exp = explainer.explain_instance(text, prediction.predictor, labels=(0, 1, 2, 3, 4), num_features=5,
                                 num_samples=len(text.split()))
exp.show_in_notebook(text=True)

# %%

exp.save_to_file('Dataset/Amazon Food Reviews/html/label3.html')

# %%

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


# %%
def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(
        dtype=next(model_bert.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(
            dtype=next(model_bert.parameters()).dtype)  # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                  1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertModelWrapper(nn.Module):

    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, embeddings):
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)


bert_model_wrapper = BertModelWrapper(model_B)
ig = IntegratedGradients(bert_model_wrapper)

# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []


def interpret_sentence(model_wrapper, sentence, label=1):
    model_wrapper.eval()
    model_wrapper.zero_grad()

    input_ids = torch.tensor([tokenizer_B.encode(sentence, add_special_tokens=True)])
    input_embedding = model_wrapper.model.bert.embeddings(input_ids)

    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)

    print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))

    tokens = tokenizer_B.convert_ids_to_tokens(input_ids[0].numpy().tolist())
    add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label, delta, vis_data_records_ig)


def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred,
        pred_ind,
        label,
        "label",
        attributions.sum(),
        tokens[:len(attributions)],
        delta))


# %%

interpret_sentence(bert_model_wrapper, sentence=df_AR.iloc[26, 1], label=df_AR.iloc[26, 0])
visualization.visualize_text(vis_data_records_ig)

# %%
