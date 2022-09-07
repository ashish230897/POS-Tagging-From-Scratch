
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from nltk.corpus import brown
import re

def compute_emission(tags, words, tag_words):
  emission = defaultdict()
  for tag in tags.keys():
      for word in words.keys():  
          if tag not in emission:
              emission[tag] = defaultdict()
          count = 0
          if word in tag_words[tag]: count = tag_words[tag][word]
          else: count=0.00000001
          emission[tag][word] = count/tags[tag]
  return emission

def compute_transition(tags, train_sent):
  sents = []
  for lis in train_sent:
    sents.append(lis)
  count = 0
  for sent in sents:
    if count > 10: break
    if len(sent) == 5: 
        count += 1
  
  bigram = defaultdict()
  for sent in sents:
    length = len(sent)
    for i in range(0, length-1):
        curr = sent[i][1]
        next_ = sent[i+1][1]

        bi = curr + '-' + next_
        if bi not in bigram: bigram[bi] = 1
        else: bigram[bi] += 1
  
  for sent in sents:
    start_tag = sent[0][1]
    bi = "^-" + start_tag
    if bi not in bigram: bigram[bi] = 1
    else: bigram[bi] += 1
  
  transition = defaultdict()
  for tag in tags.keys():
    for tag1 in tags.keys():
        if tag not in transition:
            transition[tag] = defaultdict()
        if tag+'-'+tag1 in bigram:
          transition[tag][tag1] = (bigram[tag+'-'+tag1]+1)/(tags[tag]+len(tags))
          #transition[tag][tag1] = (bigram[tag+'-'+tag1]+1)/(tags[tag]+len(tags))
        else:
          transition[tag][tag1] = 1/(tags[tag]+len(tags));
          #transition[tag][tag1] = 0.0001;
  
  transition["^"] = defaultdict()
  for tag in tags.keys():
     transition["^"][tag] = bigram["^"+"-"+tag]/len(sents)
  
  return transition

def compute_param(train_word, train_sent):
  tags = defaultdict()
  words = defaultdict()
  tag_words = defaultdict()
  
  for tuple in train_word:
      tag = tuple[1]
      word = tuple[0]
      if tag in tags: tags[tag] += 1
      else: tags[tag] = 1
        
      if word in words: words[word] += 1
      else: words[word] = 1
        
      if tag in tag_words:
          if word in tag_words[tag]: tag_words[tag][word] += 1
          else: tag_words[tag][word] = 1
      else:
          tag_words[tag] = defaultdict()
          tag_words[tag][word] = 1
  
  emission = compute_emission(tags, words, tag_words)
  transition = compute_transition(tags, train_sent)
  
  parameters = {"emission": emission, "transition": transition, "tags": tags};
  return parameters