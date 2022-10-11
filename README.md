# POS-Tagging-From-Scratch
Here we have implemented POS Tagging using 3 techniques:  
1. HMM Viterbi algorithm  
2. HMM Viterbi with word embeddings to handle unknown words well.  
3. FeedForward Neural Network trained on word embedding feature vectors.  

Dependencies and Libraries used are :- 
NLTK , numpy , pandas , matplotlib , collections , copy , re (regular expression), Pytorch, Word2Vec.  
Dataset used :- NLTK Corpus brown universal tagged set.

## Setting up the environment:
Run the command 'conda create --name <env> --file requirements.txt' which will use requirements.txt file to setup conda environment.

## Training the model for different settings:  
### Training HMM Viterbi Algorithm:  
To train the HMM model run **train.py** file which will train the model on full brown dataset and saves the training parameters in parameters.pkl file.  
### Train Neural Network:  
To train the nn model, change the parameter fold in the file train_nn.py to train the neural network on the 4 folds excluding the one fold selected, then run the file as: `python train_nn.py` 

## Testing the model on sample sentences
Run test.py file using the command: `python test.py -i "He has a car"`.  
To use the word embedding, use the command: `python test.py -i "He has a car" -w True`.  
To use the neural network, use the command: `python test_nn.py -i "He has a car"`.

## K Fold Analysis on Brown Corpus for the HMM Viterbi Algorithm
kFoldAnalysis.py is to be used to train and test data using 5 fold cross validation method.  
Accuracy is visualized using confusion matrix heatmap, also the file prints overall F1 score and F1 score per tag.
Run kFoldAnalysis.py file using the command: `python kFoldAnalysis.py -w True` to do an analysis with the word embedding model and use `python kFoldAnalysis.py` to do an analysis without it.