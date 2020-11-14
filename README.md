# Capstone

The [Code skeleton](https://github.com/fajim1/Capstone/tree/master/Code%20Skeleton) directory gives a general gives a general structure to use on any multilabel text dataset while 
[Dataset + Code](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code) directory is dataset specific. Right now only the [Amazon Food Reviews](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews) is complete with the other datasets being updated soon enough. This Readme will go through the [Amazon Food Reviews](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews) directory so that it can be reproduced 

# General structure 

There are 6 python scripts which needs to be run in a specific order (given below) to ensure reproducibility. There are also four sub-directories for convenience, namely to store the original dataset, pre processed dataset, trained model and visualizations. Keeping the format of the sub-directories is optional but the path to load the models and dataset should be carefully maintained in the .py scripts. 

### Note : 
Anywhere that the path to a model or dataset is required it is marked in the .py files as a #comments

# 1 preprocess.py 

The general preprocess structure is keeping only the text and label column and undersampling the majority class. The preprocessed dataset goes to [processed_data](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/processed_data) directory or saved in a directory of choosing and needs to be loaded in the next Train.py script 

# 2 Train.py

*The models are also stored in google storage so this script is optional. The wget url is inside the next scripts*

This script requires to load the preprocessed dataset from the previous script and train the models on it. The models are then saved in  [model](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/model) directory or a directory of choosing and needs to be loaded in the next Predict.py script. The models I trained is saved in google storage 

# 3 Predict.py 

This Script requires the model to be loaded from either google storage (The wget url is inside the script) or from the local directory. The script also requires to load the preprocessed dataset. The predicted and true labels are saved as a dataset in [processed_data](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/processed_data) directory or a directory of choosing and needs to be loaded in the next Evaluate.py script. The purpose of this script and next (Evaluate.py) is to check the performance of the model before looking at its interpretability 

# 4 Evaluate.py

This script requires to load the dataset with true and predicted labels from the previous scripts. The confusion matrix and classification report for each model is outputted in this script 

# 5 Visualize_lime.py and Visualize_captum.py

These Scripts requires the model to be loaded from either google storage(The wget url is inside the script) or from the local directory. Here you need to pass your own example texts or texts from the dataset (in this case the script requires to load the preprocessed dataset). The output from lime is saved in [html](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/html) or a directory of choosing
Note - At the moment, Captum only works with Bert and will be worked on to see if it works with Roberta or Albert. 




