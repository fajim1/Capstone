# Capstone

The [Code skeleton](https://github.com/fajim1/Capstone/tree/master/Code%20Skeleton) directory gives a general structure to use on any Binary text dataset while 
[Dataset + Code](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code) directory is dataset specific.  This Readme will go through the [Amazon Food Reviews](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Restaurant%20Reviews) directory so that it can be reproduced 

# General structure 

There are 6 python scripts which needs to be run in a specific order (given below) to ensure reproducibility. There are also three sub-directories for convenience, namely to store the original dataset, pre processed dataset and visualizations. Keeping the format of the sub-directories is optional but the path to load the models and dataset should be carefully maintained in the .py scripts. 

### Note : 
Anywhere that the path to a model or dataset is required is marked in the .py files as a #comments

# 1 preprocess.py 

The general preprocess structure is keeping only the text and label column and undersampling the majority class. The preprocessed dataset goes to [processed_data](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/processed_data) directory or saved in a directory of choosing and needs to be loaded in the next Train.py script 

# 2 Train.py


This script requires to load the preprocessed dataset from the preprocess.py script and train the BERT model on it. The models are then saved in  [model](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/model) directory or a directory of choosing and needs to be loaded in the next Predict.py script. The models I trained is saved in google storage 

# 3 Train_CNN.py

This script This script requires to load the preprocessed dataset from the preprocess.py script and trains the baseline CNN model as well as visualize the results of the cnn model in this script. Glove embeddings are needed for this dataset training which is provided in the code using wget. Note: To see the visualizations, the scripts must be run in a notebook like jupyter or google colab in order to see the visualization.


# 4 Visualize_lime.py 

These Scripts requires the model to be loaded from either google storage(The wget url is inside the script) or from the local directory. Here you need to pass your own example texts or texts from the dataset (in this case the script requires to load the preprocessed dataset). The output from lime is saved in [html](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/html) or a directory of choosing
Note: To see the visualizations, the scripts must be run in a notebook like jupyter or google colab in order to see the visualization.

# 5 Visualize_captum.py

These Scripts requires the model to be loaded from either google storage(The wget url is inside the script) or from the local directory. Here you need to pass your own example texts or texts from the dataset (in this case the script requires to load the preprocessed dataset). The output from lime is saved in [html](https://github.com/fajim1/Capstone/tree/master/Dataset%20%2B%20Code/Amazon%20Food%20Reviews/html) or a directory of choosing
Note: To see the visualizations, the scripts must be run in a notebook like jupyter or google colab in order to see the visualization.


