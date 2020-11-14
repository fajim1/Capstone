# Capstone

The [Code skeleton](https://github.com/fajim1/Capstone/tree/master/Code%20Skeleton) directory gives a general gives a general structure to use on any multilabel text dataset while 
[Dataset + Code](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code) directory is dataset specific. Right now only the [Amazon Food Reviews](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews) is complete with the other datasets being updated soon enough. This Readme will go through the [Amazon Food Reviews](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews) directory so that it can be reproduced 

# General structure 

There are 6 python scripts which needs to be run in a specific order to ensure reproducibility. There are also four sub-directories for convenience, namely to store the original dataset, pre processed dataset, trained model and visualizations. Keeping the format of the sub-directories is optional but the path to load the models and dataset should be carefully maintained in the .py scripts. 

### Note : 
Anywhere that the path to a model or dataset is required is marked in the .py files in #comments

# 1 preprocess.py 

The general preprocess structure is keeping only the text and label column and undersampling the majority class. The preprocessed dataset is saved and needs to be loaded in the next Train.py script 

# 2 Train.py

This script 

