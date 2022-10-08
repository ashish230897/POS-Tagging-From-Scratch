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
from gensim.models import Word2Vec
import gensim
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('punkt')
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
    
    parameters = compute_param(train_words, train_sent, vocab)
    parameters['vocab'] = vocab
    
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(parameters, f)
        
    # train word2vec and save its parameters
    train_set = brown.sents()
    book1 = nltk.corpus.gutenberg.sents('austen-emma.txt')
    book2 = nltk.corpus.gutenberg.sents('austen-persuasion.txt')
    book3 = nltk.corpus.gutenberg.sents('austen-sense.txt')
    book4 = nltk.corpus.gutenberg.sents('bible-kjv.txt')
    book5 = nltk.corpus.gutenberg.sents('blake-poems.txt')
    book6 = nltk.corpus.gutenberg.sents('bryant-stories.txt')
    book7 = nltk.corpus.gutenberg.sents('burgess-busterbrown.txt')
    book8 = nltk.corpus.gutenberg.sents('burgess-busterbrown.txt')
    book9 = nltk.corpus.gutenberg.sents('carroll-alice.txt')
    book10 = nltk.corpus.gutenberg.sents('chesterton-ball.txt')
    book11 = nltk.corpus.gutenberg.sents('chesterton-thursday.txt')
    book12 = nltk.corpus.gutenberg.sents('edgeworth-parents.txt')
    book13 = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')
    book14 = nltk.corpus.gutenberg.sents('milton-paradise.txt')
    book15 = nltk.corpus.gutenberg.sents('shakespeare-caesar.txt')
    book16 = nltk.corpus.gutenberg.sents('shakespeare-hamlet.txt')
    book17 = nltk.corpus.gutenberg.sents('shakespeare-macbeth.txt')
    book18 = nltk.corpus.gutenberg.sents('whitman-leaves.txt')
    big_vocab = book1 + book2 + book3 + book4 + book5 + book6 + book7 + book8 + book9 + book10 + book11 + book12 + book13 + book14 + book15 + book16 + train_set 
    
    print("Training word2vec model")
    model = gensim.models.Word2Vec(big_vocab, min_count=1)
    model.save('big.embedding')

    big = {}
    for sent in big_vocab:
        for token in sent:
            if token not in big: big[token] = 1
            else: continue

    word_vec = {}
    for word in big:
        word_vec[word] = model.wv[word]
    
    with open('wordVector.pkl', 'wb') as f:
        pickle.dump(word_vec, f)

def main():
    # train on entire data
    train_model()

if __name__ == "__main__":
    main()
    
