import nltk
import numpy as np
import pandas as pd
import copy
from nltk.corpus import brown
from gensim.models import Word2Vec
import gensim
import gc
import os
import random
import torch
nltk.download('brown')
nltk.download('universal_tagset')
import gensim.downloader as api

import warnings
warnings.simplefilter('ignore')
from nn_train_utils import *
from viterbi_algo import *
import pickle


# Perform inference on a sentence
def inference(sent, word_model, parameters, index_tag, tag_index):
    net = Net(parameters)
    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    net.to(device)
    
    checkpoint_path = parameters["OUT_DIR"]
    net.load_state_dict(torch.load(checkpoint_path + "nn_model.pt", map_location=torch.device('cpu')))
    net.eval()
    
    X_sentence = []
    for word in sent.strip().split():
        X_sentence.append(word)
    
    X = []
    absent = []
    for i,word in enumerate(X_sentence):
        X_triplet = []
        vector_curr = []
        vector_prev = []
        vector_prevv = []
        
        if X_sentence[i] in word_model:
            vector_curr = list(word_model[X_sentence[i]])
        else:
            absent.append(i)
            continue
        
        if i != 0 and X_sentence[i-1] in word_model:
            vector_prev = list(word_model[X_sentence[i-1]])
        else:
            vector_prev = np.zeros(300)
        
        if i != 0 and i != 1 and X_sentence[i-2] in word_model:
            vector_prevv = list(word_model[X_sentence[i-2]])
        else:
            vector_prevv = np.zeros(300)
        
        X_triplet.extend(vector_curr)
        X_triplet.extend(vector_prev)
        X_triplet.extend(vector_prevv)
        
        X = X + [X_triplet]
    
    if len(X) < 1: return []
    X_tensor = torch.FloatTensor(X)
    X_tensor = X_tensor.to(device)

    outputs = net(X_tensor)
    states = []
    for i,output in enumerate(outputs):
        index = max_(output)
        states.append(index_tag[index])
    
    if len(absent) > 0:
        obj = Viterbi()
        # load parameters
        with open('parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
        states_viterbi = obj.compute_states(sent, parameters, word_model, False)
        print("Viterbi states are {}".format(states_viterbi))
    

    for index in absent:
        states.insert(index, states_viterbi[index+1])
    
    return states


def main():
    sent = "Sameer is a good boy"

    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

    word_model = api.load("word2vec-google-news-300")
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
        "hidden_size3": 500,
        "num_classes": 12,
        "num_epochs": 9,
        "batch_size": 50,
        "learning_rate": 1e-6,
        "OUT_DIR": "./",
    }

    states = inference(sent, word_model, train_parameters, index_tag, tag_index)
    print("States for sentence {} is {}".format(sent, states))



if __name__ == "__main__":
    main()