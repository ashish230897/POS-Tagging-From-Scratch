import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from nltk.corpus import brown
import re

# storing all the POS tags, words
class TreeNode:
  def __init__(self, tag, prob, parent):
      self.tag = tag
      self.prob = prob
      self.tags = []
      self.parent = parent
        

class Viterbi:

    def find_tags(self, imp_nodes):
        max_node = None
        for tag,tag_node in imp_nodes.items():
            if max_node is None or max_node.tags[0].prob < tag_node.tags[0].prob: max_node = tag_node
     
        tags = []
        while(max_node is not None):
            tags.insert(0, max_node.tag)
            max_node = max_node.parent
     
        return tags
     
    def compute_states(self, sent, parameters):
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
            for tag,tag_node in imp_nodes.items():
              # compute every tag for this node
                if i == len(tokens)-1:
                    if token in parameters["vocab"]:
                          emission = parameters["emission"][tag][token]
                    else:
                      emission = 0.001
                    transition = parameters["transition"][tag]["."]
                    new_prob = tag_node.prob*emission*transition
                    child = TreeNode(".", new_prob, tag_node)
                    tag_node.tags.append(child)
                else:
                    for child_tag in parameters["tags"].keys():
                        if token in parameters["vocab"]:
                          emission = parameters["emission"][tag][token]
                        else:
                          emission = 0.001
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