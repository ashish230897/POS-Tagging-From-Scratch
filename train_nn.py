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

def evaluate_model(parameters, tagged_sents, fold, word_model, tag_index):
    net = Net(parameters)
    net.load_state_dict(torch.load(parameters["OUT_DIR"] + "NN_POS_Tagging/best_checkpoint/nn_model.pt"))
    print("Computing validation accuracy")
    compute_accuracy(net, tagged_sents, fold, word_model, tag_index)
    print("Computing train accuracy")
    compute_accuracy(net, tagged_sents, fold, word_model, tag_index)

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

    # train word2vec model
    #word_model = Word2Vec(brown_sents, min_count=1, vector_size=300, epochs=20)
    #word_model.save('new.embedding')
    #word_model = Word2Vec.load("./new.embedding")

    UNIVERSAL_TAGS = ["VERB","NOUN","PRON","ADJ","ADV","ADP","CONJ","DET","NUM","PRT","X","."]
    
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

    # iterate over the sentences and train the model on the fly
    #model = train_sents(train_parameters, tagged_sents, word_model, 0, tag_index, index_tag)

    # load pre-trained model and pass it for computing accuracy on validation set
    #evaluate_model(train_parameters, tagged_sents, 0, word_model, tag_index)

    gc.collect()


if __name__ == "__main__":
    main()