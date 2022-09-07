import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from nltk.corpus import brown
import re
from compute_params import *
from viterbi_algo import *
import pickle

nltk.download('brown')
nltk.download('universal_tagset')
l = brown.tagged_sents(tagset='universal')

def train_model():
    train_sent = l
    train_words = []
    
    vocab = {}
    for sent in train_sent:
        for tup in sent:
            train_words.append(tup)
            if tup[0] in vocab: vocab[tup[0]] += 1
            else: vocab[tup[0]] = 1
    
    parameters = compute_param(train_words,train_sent)
    parameters['vocab'] = vocab
    
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(parameters, f)
    

def main():
    # train on entire data
    train_model()

if __name__ == "__main__":
    main()