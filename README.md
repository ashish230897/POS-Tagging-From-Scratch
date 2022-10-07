# POS-Tagging-From-Scratch
Here we have implemented POS Tagging using HMM Viterbi algorithm.

Dependencies and Libraries used are :- 
NLTK , numpy , pandas , matplotlib , collections , copy , re (regular expression).  
Dataset used :- NLTK Corpus brown universal tagged set.

## Setting up the environment:
Run the command 'conda create --name <env> --file requirements.txt' which will use requirements.txt file to setup conda environment.

## Training the model on entire corpus
To train the HMM model run **train.py** file which will train the model on full dataset and saves the training parameters in parameters.pkl file.  
We also train a **Word2Vec** model in this file, this model is used in the following way:  
1. If the HMM algorithm encounters an unknown word, the unknown word is vectorised using word2vec.  
2. This vector is then compared(using cosine-similarity) with all the words of the training(Brown) corpus to find the most similar word present in the corpus.  
3. This similar word then can be used to find the emission probability.

## Testing the model on sample sentences
Run test.py file using the command: `python test.py -i "He has a car"`.  
To use the word embedding, use the command: `python test.py -i "He has a car" -w True`.

## K Fold Analysis on Brown Corpus
kFoldAnalysis.py is to be used to train and test data using 5 fold cross validation method.  
Accuracy is visualized using confusion matrix heatmap, also the file prints overall F1 score and F1 score per tag.
Run kFoldAnalysis.py file using the command: `python test.py -w True` to do an analysis with the word embedding model and use `python test.py` to do an analysis without it.