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

def test_data(input, parameters):
    obj = Viterbi()
    
    sent = " ".join(input.split())
    states = obj.compute_states(sent, parameters)
    print("States corresponding to input are: ", sent, states)

def main():
    # collect the input sentence
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--Input", help = "Provide Input")
    args = parser.parse_args()

    # load parameters
    with open('parameters.pkl', 'rb') as f:
        parameters = pickle.load(f)

    if args.Input == None:
        print("Please provide input")
    else:
        test_data(args.Input, parameters)
    

if __name__ == "__main__":
    main()