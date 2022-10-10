import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from nltk.corpus import brown
import re
from gensim.models import Word2Vec
import gensim
import gc
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import random
import seaborn as sns
import gensim.downloader as api
from gensim.models import Word2Vec
nltk.download('brown')
nltk.download('universal_tagset')

import warnings
warnings.simplefilter('ignore')

from nn_train_utils import *

def evaluate_model(parameters, tagged_sents, fold, word_model, tag_index, acc_per_tag, confusion_matrix):
    net = Net(parameters)
    net.load_state_dict(torch.load(parameters["OUT_DIR"] + "nn_model.pt"))
    print("Computing validation accuracy")
    compute_accuracy(net, tagged_sents, fold, word_model, tag_index, acc_per_tag, confusion_matrix)

def main():
    # making model deterministic
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tagged_sents = brown.tagged_sents(tagset='universal') # list of list of tagged tokens
    brown_sents = brown.sents() # list of list of tokens
    
    word_model = api.load("word2vec-google-news-300")
    confusion_matrix = {}
    acc_per_tag = {}

    UNIVERSAL_TAGS = ["VERB","NOUN","PRON","ADJ","ADV","ADP","CONJ","DET","NUM","PRT","X","."]

    for tag in UNIVERSAL_TAGS:
        confusion_matrix[tag] = {}
        for pos in UNIVERSAL_TAGS:
            confusion_matrix[tag][pos] = 0

    for fold in range(1,2):
        acc_per_tag[fold] = {}
        for tag in UNIVERSAL_TAGS:
            acc_per_tag[fold][tag] = {}
            acc_per_tag[fold][tag]['TP'] = 0
            acc_per_tag[fold][tag]['FP'] = 0
            acc_per_tag[fold][tag]['FN'] = 0
    
    count = 0
    tag_index = {}
    for tag in UNIVERSAL_TAGS:
        if tag not in tag_index:
            tag_index[tag] = count
            count += 1

    index_tag = {}
    for tag,value in tag_index.items():
        index_tag[value] = tag

    print(tag_index)
    print(index_tag)

    train_parameters = {
        "input_size": 900,
        "hidden_size1": 1200,
        "hidden_size2": 800,
        "num_classes": 12,
        "num_epochs": 9,
        "batch_size": 50,
        "learning_rate": 1e-6,
        "OUT_DIR": "./",
    }
    split_arr = np.array_split(tagged_sents, 5)

    for fold in range(1,2):
        sents = []
        for i in range(5):
            if i == fold: continue
            sents += list(split_arr[i])
        
        model = train_sents(train_parameters, sents, word_model, fold, tag_index, index_tag)
        # load pre-trained model and pass it for computing accuracy on validation set
        evaluate_model(train_parameters, list(split_arr[fold]), fold, word_model, tag_index, acc_per_tag, confusion_matrix)

    gc.collect()


if __name__ == "__main__":
    main()