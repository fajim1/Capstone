import sklearn
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

#%%

df_AR = pd.read_csv('Dataset/Amazon Food Reviews/processed_data/Preprocess.csv')


sns.countplot(df_AR['Score'])
plt.show()

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 4,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(df_AR['Summary'])


#%%
import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='poly',probability=True)

classifier_linear.fit(train_vectors, df_AR['Score'])

prediction_linear = classifier_linear.predict(train_vectors)

#%%
# results
y_test = df_AR['Score']
y_pred = prediction_linear

print("SVM")
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
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3','Class 4','Class 5']))
print('\n')

#%%
import pickle
# save the model to disk
pickle.dump(classifier_linear, open('Dataset/Amazon Food Reviews/model/svm.pkl', 'wb'))

#%%
# load the model from disk
loaded_model = pickle.load(open('Dataset/Amazon Food Reviews/model/svm.pkl', 'rb'))
result = loaded_model.predict(train_vectors)
print(result)