# POS-Tagging-From-Scratch
Here we have implemented POS Tagging using HMM Viterbi algorithm.

Dependencies and Libraries used are :- 
NLTK , numpy , pandas , matplotlib , collections , copy , re (regular expression).  
Dataset used :- NLTK Corpus brown universal tagged set.

## Setting up the environment:
Run the command 'conda create --name <env> --file requirements.txt' which will use requirements.txt file to setup conda environment.

## Training the model on entire corpus
To train the HMM model run **train.py** file which will train the model on full dataset and saves the training parameters in parameters.pkl file.

## Testing the model on sample sentences
Run test.py file using the command: `python test.py -i "He has a car"`.

## K Fold Analysis on Brown Corpus
kFoldAnalysis.py is to be used to train and test data using 5 fold cross validation method.  
Accuracy is visualized using confusion matrix heatmap, also the file prints overall F1 score and F1 score per tag.
