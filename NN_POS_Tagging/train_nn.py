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
from gensim.models import Word2Vec
nltk.download('brown')
nltk.download('universal_tagset')

import warnings
warnings.simplefilter('ignore')

from nn_train_utils import *

def evaluate_model(parameters, train_dataloader, val_dataloader):
    net = Net(parameters)
    net.load_state_dict(torch.load(parameters["OUT_DIR"] + "NN_POS_Tagging/best_checkpoint/nn_model.pt"))
    print("Computing validation accuracy")
    compute_accuracy(net, val_dataloader)
    print("Computing train accuracy")
    compute_accuracy(net, train_dataloader)

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

    # train word2vec model
    word_model = Word2Vec(brown_sents, min_count=1, vector_size=300, epochs=20)
    word_model.save('new.embedding')
    word_model = Word2Vec.load("./new.embedding")

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
        "input_size": 240,
        "hidden_size1": 700,
        "hidden_size2": 300,
        "num_classes": 12,
        "num_epochs": 15,
        "batch_size": 50,
        "learning_rate": 1e-6,
        "OUT_DIR": "./",
    }

    # iterate over the sentences and train the model on the fly
    # for epochs in range(train_parameters["num_epochs"]):
    #     for sent in tagged_sents:
    #         # pass this tagged sentence for training

    train_dataloader, val_dataloader = prepare_fold_data(0, tagged_sents, tag_index, index_tag, word_model, train_parameters)
    model = train_model(train_dataloader, val_dataloader, train_parameters)

    # load pre-trained model and pass it for computing accuracy on validation set
    evaluate_model(train_parameters, train_dataloader, val_dataloader)

    gc.collect()


if __name__ == "__main__":
    main()