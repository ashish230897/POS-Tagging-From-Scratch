import torch
import torch.nn as nn
import nltk
import numpy as np
import pandas as pd
import copy
from nltk.corpus import brown
import re
import gensim
import gc
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, parameters):
        super(Net, self).__init__()     
        self.fc1 = nn.Linear(parameters["input_size"], parameters["hidden_size1"])
        self.gelu = nn.GELU()
        
        self.fc2 = nn.Linear(parameters["hidden_size1"], parameters["hidden_size1"])
        self.fc3 = nn.Linear(parameters["hidden_size1"], parameters["hidden_size2"])

        self.fc4 = nn.Linear(parameters["hidden_size2"], parameters["num_classes"])
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)

        out = self.fc3(out)
        out = self.gelu(out)
        
        out = self.fc4(out)
        return out


def max_(out):
    """
    Find the maximum element index from out list.

    """
    index = 0
    maximum = max(out).item()
    for i,ele in enumerate(out):
        if ele.item() == maximum: 
            index = i
            break
    return index
    

def compute_accuracy(model, tagged_sents, fold, word_model, tag_index, acc_per_tag, confusion_matrix):
    valid_data = tagged_sents
    UNIVERSAL_TAGS = ["VERB","NOUN","PRON","ADJ","ADV","ADP","CONJ","DET","NUM","PRT","X","."]

    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    model.to(device)

    accuracies = []
    model.eval()

    for eval_sent in valid_data:
        Y, X = input_fun(eval_sent, word_model)
        if len(X) < 1: continue
        X_tensor = torch.FloatTensor(X)
        X_tensor = X_tensor.to(device)

        Y = [tag_index[y] for y in Y]

        Y_tensor = torch.LongTensor(Y)
        Y_tensor = Y_tensor.to(device)

        outputs = model(X_tensor)
        
        for i,output in enumerate(outputs):
            index = max_(output)
            actual = Y_tensor[i].item()
            index_pos = UNIVERSAL_TAGS[index]
            actual_pos = UNIVERSAL_TAGS[actual]
            confusion_matrix[actual_pos][index_pos] += 1
            if index == actual:
                acc_per_tag[fold][index_pos]['TP'] += 1
                accuracies.append(1)
            else:
                acc_per_tag[fold][index_pos]['FP'] += 1
                acc_per_tag[fold][actual_pos]['FN'] += 1
                accuracies.append(0)

    accuracy = sum(accuracies)/len(accuracies)
    print("Validation data accuracy is {}".format(accuracy))

def input_fun(sentence, word_model):

    X = [] 
    Y = [] 
    X_sentence = []
    Y_sentence = []
    
    for word in sentence:
        X_sentence.append(word[0]) # entity[0] contains the word
        Y_sentence.append(word[1])
    for i,word in enumerate(X_sentence):
        X_triplet = []
        vector_curr = []
        vector_prev = []
        vector_prevv = []
        
        if X_sentence[i] in word_model:
            vector_curr = list(word_model[X_sentence[i]])
        else: continue
        
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
        Y = Y + [Y_sentence[i]]
    
    return Y, X

def evaluate(model, criterion, valid_data, word_model, tag_index, device):
    
    losses = []
    model.eval()
    
    batches = 0
    for eval_sent in valid_data:
        if batches == 1000: break
        Y, X = input_fun(eval_sent, word_model)
        if len(X) < 1: continue
        X_tensor = torch.FloatTensor(X)
        X_tensor = X_tensor.to(device)

        Y = [tag_index[y] for y in Y]

        Y_tensor = torch.LongTensor(Y)
        Y_tensor = Y_tensor.to(device)

        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        losses.append(loss.item())
        batches += 1

    avg_loss = sum(losses)/len(losses)
    print("Validation data loss is {}".format(avg_loss))
    
    return avg_loss


def train_sents(parameters, tagged_sents, word_model, fold, tag_index, index_tag):
    
    split_data = np.array_split(tagged_sents, 5)
    valid_data = list(split_data[fold])
    
    train_data = []
    for j in range(0,5):
        if(j == fold): continue
        train_data += list(split_data[j])
    
    # selecting a random subset to use less resources
    random.seed(42)
    train_data = random.choices(train_data, k = int(0.6*len(train_data)))

    net = Net(parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=parameters["learning_rate"])
    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")
    net.to(device)

    steps = 0
    last_loss = 1000
    checkpoint_path = parameters["OUT_DIR"]
    print("Total steps are {}".format(len(train_data)*parameters["num_epochs"]))

    for epochs in range(parameters["num_epochs"]):
        for sent in train_data:
            # pass this tagged sentence for training
            net.train()
            Y, X = input_fun(sent, word_model)
            if len(X) < 1: continue
            X_tensor = torch.FloatTensor(X)
            X_tensor = X_tensor.to(device)
            Y = [tag_index[y] for y in Y]

            Y_tensor = torch.LongTensor(Y)
            Y_tensor = Y_tensor.to(device)

            outputs = net(X_tensor)
            loss = criterion(outputs, Y_tensor)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            steps += 1
            if steps % 2000 == 0:
                print("Train loss upto step {} is {}".format(steps, loss.item()))
                loss = evaluate(net, criterion, valid_data, word_model, tag_index, device)
                if last_loss > loss:
                    last_loss = loss
                    # save model weights
                    torch.save(net.state_dict(), checkpoint_path + "nn_model{}.pt".format(fold))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    
    loss = evaluate(net, criterion, valid_data, word_model, tag_index, device)
    if last_loss > loss:
        # save model weights
        torch.save(net.state_dict(), checkpoint_path + "nn_model.pt")
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

    return net


def create_heatmap(acc_per_tag, confusion_matrix, fold):
    arr = []
    keys = list(confusion_matrix.keys())
    #keys.remove("^")
    for key in keys:
        if key == "^": continue
        lis = []
        dict_ = confusion_matrix[key]
        for key_ in keys:
            if key_ == "^": continue
            lis.append(confusion_matrix[key][key_])
        arr.append(lis)
    arr = np.array(arr)

    fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
    plot = sns.heatmap(arr, annot=True, linewidths=.5, ax=ax, xticklabels=keys, yticklabels=keys, fmt='g')
    fig = plot.get_figure()
    fig.savefig("./heatmap{}.png".format(fold))