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

nltk.download('brown')
nltk.download('universal_tagset')

# global params
accuracy = {}
confusion_matrix = {}
acc_per_fold = {}
cnf_per_fold = {}

# split train data in 5 parts
l = brown.tagged_sents(tagset='universal')
split_arr = np.array_split(l,5)

def compute_per_tag_acc():
    f1_per_tag= {}
    pre_per_tag = {}
    rec_per_tag = {}
    for key in acc_per_fold[0]:
        f1_per_tag[key] = 0
        pre_per_tag[key] = 0
        rec_per_tag[key] = 0
        
    for fold in acc_per_fold:
        for tag in acc_per_fold[fold]:
            pre = acc_per_fold[fold][tag]['TP']/(acc_per_fold[0][tag]['TP']+acc_per_fold[0][tag]['FP'])
            pre_per_tag[tag] += pre
            rec = acc_per_fold[fold][tag]['TP']/(acc_per_fold[0][tag]['TP']+acc_per_fold[0][tag]['FN'])
            rec_per_tag[tag] += rec 
            f1_per_tag[tag] += round((2*pre*rec/(pre+rec)),2)
            
    for key in f1_per_tag:
        f1_per_tag[key] = round(f1_per_tag[key]/5,2)
        pre_per_tag[key] = round(pre_per_tag[key]/5,2)
        rec_per_tag[key] = round(rec_per_tag[key]/5,2)

    print("Per tag F1 score is:", f1_per_tag)
    print("Per tag precision is:", pre_per_tag)
    print("Per tag recall is:", rec_per_tag)


def acc(prediction,true_label,accuracy):  
  
  for i in range(0,len(true_label)):
    confusion_matrix[true_label[i]][prediction[i]] += 1
    
    if(prediction[i]==true_label[i]):
      accuracy[prediction[i]]['TP']+=1
    else:
      accuracy[prediction[i]]['FP']+=1
      accuracy[true_label[i]]['FN']+=1

def test_data(i):
    test = split_arr[i]
    X_test = []
    Y_test = []
    for sent in test:
        test_sent = ""
        tag_test = []
    
        for tup in sent:
            test_sent += tup[0] + " ";
            test_sent.strip()
            tag_test.append(tup[1])
        Y_test.append(['^'] + tag_test)
        X_test.append(test_sent)
    return (X_test,Y_test)

def init_cnf_matrix(parameters):
    for key_1 in parameters['tags'].keys():
        confusion_matrix[key_1] = {}
        for key_2 in parameters['tags'].keys():
            confusion_matrix[key_1][key_2] = 0
            confusion_matrix[key_1]['^'] = 0
    confusion_matrix['^'] = {}
    for key in parameters['tags']:
        confusion_matrix['^'][key] = 0
    confusion_matrix['^']['^'] = 0

def test_acc(X_test,Y_test,parameters):
    
    obj = Viterbi()
    size = len(X_test)
    
    for i in range(0,size):
        sent = " ".join(X_test[i].split())
        if(len(sent) > 0):
            states = obj.compute_states(sent, parameters)
            acc(states,Y_test[i],accuracy)
    return (accuracy)

def acc_per_tag(confusion_matrix):
    accuracy_per_tag = {}
    for key in confusion_matrix:
        accuracy_per_tag[key] = confusion_matrix[key][key]/(1+sum(confusion_matrix[key].values()))
        accuracy_per_tag[key] = round(accuracy_per_tag[key],2)
    return (accuracy_per_tag)

def init_acc(parameters):
    for key in parameters['tags']:
        accuracy[key] = {}
        accuracy[key]['TP'] = 0
        accuracy[key]['FP'] = 0
        accuracy[key]['FN'] = 0
    
    accuracy['^'] = {}
    accuracy['^']['TP'] = 0
    accuracy['^']['FP'] = 0
    accuracy['^']['FN'] = 0

def train_and_test(i):
    train_sent = []
    train_words = []
    for j in range(0,5):
        if(j!=i):
            train_sent += list(split_arr[j])
            vocab = {}
            for sent in train_sent:
              for tup in sent:
                train_words.append(tup)
                if tup[0] in vocab: vocab[tup[0]] += 1
                else: vocab[tup[0]] = 1
    
    parameters = compute_param(train_words,train_sent)
    parameters['vocab'] = vocab
    
    init_cnf_matrix(parameters)
    init_acc(parameters)
    
    # preparing test data
    X_test,Y_test = test_data(i)
    
    acc_per_fold[i] = {}
    acc_per_fold[i] = test_acc(X_test,Y_test,parameters)
    
    print(acc_per_tag(confusion_matrix) , i+1)

def create_heatmap():
    arr = []
    keys = list(confusion_matrix.keys())
    keys.remove("^")
    for key in keys:
        if key == "^": continue
        lis = []
        dict_ = confusion_matrix[key]
        for key_ in keys:
            if key_ == "^": continue
            lis.append(confusion_matrix[key][key_])
        arr.append(lis)
    arr = np.array(arr)

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    plot = sns.heatmap(arr, annot=True, linewidths=.5, ax=ax, xticklabels=keys, yticklabels=keys, fmt='g')
    fig = plot.get_figure()
    fig.savefig("./heatmap.png")

def compute_overall_acc():
    precision_lis = []
    recall_lis = []

    for fold in acc_per_fold:
        tp = 0 
        fp = 0 
        fn = 0 
        for tag in acc_per_fold[fold]:
            tp += acc_per_fold[fold][tag]['TP']
            fp += acc_per_fold[fold][tag]['FP']
            fn += acc_per_fold[fold][tag]['FN']
        presion = round(tp/(tp+fp),2)
        recall = round(tp/(tp+fn),2)
        precision_lis.append(presion)
        recall_lis.append(recall)

    presion = sum(precision_lis)/len(precision_lis)
    recall = sum(recall_lis)/len(recall_lis)

    print("Overall precision is" , presion)
    print("Overall recall is" , recall)

    F1 = round((2*presion*recall)/(presion+recall),2)
    print("F1 score is" , F1)
    F2 = round((5*presion*recall)/(4*presion+recall),2)
    print("F2 score is" , F2)
    F_05 = round((1.25*presion*recall)/(0.25*presion+recall),2)
    print("F 0.5 score is" , F_05)

def main():

    for i in range(0,5):
        train_and_test(i)

    # save confusion matrix heatmap for last fold
    create_heatmap()

    compute_overall_acc()

    # per tag statistics
    compute_per_tag_acc()


if __name__ == "__main__":
    main()


