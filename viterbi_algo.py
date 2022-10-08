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
from numpy.linalg import norm
from gensim import matutils
import faiss

# storing all the POS tags, words
class TreeNode:
  def __init__(self, tag, prob, parent):
      self.tag = tag
      self.prob = prob
      self.tags = []
      self.parent = parent


class Viterbi:
    
    def unknown(self, word, vocab, model, vectors, words):
        try:
            index = faiss.IndexFlatIP(100)
            index.add(vectors)
            query = np.expand_dims(model[word], axis=0)

            _, I = index.search(query, 1)
            sim_index = I[0][0]
            sim = words[sim_index]
            print("found substitute for {} is {}".format(word,sim))
            return sim

            """vector = model[word]
            max_cosine = 0
            for i in vocab.keys():
                vec_i = model[i]
                cosine = np.dot(vector,vec_i)/(norm(vector)*norm(vec_i))
                if(cosine > max_cosine):
                    max_cosine = cosine
                    sim_word = i
            return sim_word"""
        except KeyError as e:
            # word not present in word2vec vocab
            return None

    def find_tags(self, imp_nodes):
        max_node = None
        for tag,tag_node in imp_nodes.items():
            if max_node is None or max_node.tags[0].prob < tag_node.tags[0].prob: max_node = tag_node

        tags = []
        while(max_node is not None):
            tags.insert(0, max_node.tag)
            max_node = max_node.parent

        return tags

    def compute_states(self, sent, parameters, model, vectors, words, use_embedding, dict):
        sent = sent.strip()
        tokens = sent.split(' ')

        root = TreeNode("^", 1, None)
        imp_nodes = defaultdict()

        # create first level of tree
        for tag in parameters["tags"].keys():
            node = TreeNode(tag, parameters["transition"]["^"][tag], root)
            root.tags.append(node)
            imp_nodes[tag] = node

        temp_best = defaultdict()
        for i,token in enumerate(tokens):
            temp_best = defaultdict()
            cnt = True
            new = None
            for tag,tag_node in imp_nodes.items():
                # compute every tag for this node
                if i == len(tokens)-1:
                    if token in parameters["vocab"]:
                        emission = parameters["emission"][tag][token]
                    elif use_embedding:
                        if cnt:
                            if token in dict: # indexing word if it is already seen
                                new = dict[token]
                            else: new = self.unknown(token, parameters["vocab"], model, vectors, words)
                        cnt = False
                        if(new != None):
                            emission = parameters["emission"][tag][new]
                            dict[token] = new
                        else:
                            # apply laplace smoothening
                            emission = 1/(parameters["tags"][tag] + len(parameters["vocab"].keys()))
                    else: emission = 1/(parameters["tags"][tag] + len(parameters["vocab"].keys()))
                    transition = parameters["transition"][tag]["."]
                    new_prob = tag_node.prob*emission*transition
                    child = TreeNode(".", new_prob, tag_node)
                    tag_node.tags.append(child)
                else:
                    for child_tag in parameters["tags"].keys():
                        if token in parameters["vocab"]:
                            emission = parameters["emission"][tag][token]
                        elif use_embedding:
                            if cnt:
                                if token in dict: 
                                    new = dict[token]
                                else: new = self.unknown(token, parameters["vocab"], model, vectors, words)
                            cnt = False
                            if(new != None):
                                emission = parameters["emission"][tag][new]
                                dict[token] = new
                            else:
                                # apply laplace smoothening
                                emission = 1/(parameters["tags"][tag] + len(parameters["vocab"].keys()))
                        else: emission = 1/(parameters["tags"][tag] + len(parameters["vocab"].keys()))
                        transition = parameters["transition"][tag][child_tag]
                        new_prob = tag_node.prob*emission*transition
                        child = TreeNode(child_tag, new_prob, tag_node)
                        tag_node.tags.append(child)
                        if child_tag in temp_best:
                            if new_prob > temp_best[child_tag].prob:
                                temp_best[child_tag] = child
                        else:
                            temp_best[child_tag] = child

            if i < len(tokens) - 1:
                for tag in parameters["tags"].keys():
                    imp_nodes[tag] = temp_best[tag]

        return self.find_tags(imp_nodes)
    
    
