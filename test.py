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
import argparse
import pickle
from gensim.models import Word2Vec
import gensim

def test_data(input, parameters, model, use_embedding):
    obj = Viterbi()
    
    sent = " ".join(input.split())

    vectors = []
    words = []
    for word in parameters["vocab"]:
        vectors.append(model[word])
        words.append(word)
    vectors = np.array(vectors)

    states = obj.compute_states(sent, parameters, model, vectors, words, use_embedding, {})
    print("States corresponding to input are: ", sent, states)

def main():
    # collect the input sentence
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--Input", help = "Provide Input")
    parser.add_argument("-w", "--Embedding", help = "Provide Input")

    args = parser.parse_args()

    # load parameters
    with open('parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)
    
    with open('wordVector.pkl', 'rb') as f:
        model = pickle.load(f)

    if args.Input == None:
        print("Please provide input")
    else:
        test_data(args.Input, parameters, model, args.Embedding == "True")
    

if __name__ == "__main__":
    main()