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

class Net(nn.Module):
    def __init__(self, parameters):
        super(Net, self).__init__()     
        self.fc1 = nn.Linear(parameters["input_size"], parameters["hidden_size1"])
        self.gelu = nn.GELU()
        
        self.fc2 = nn.Linear(parameters["hidden_size1"], parameters["hidden_size1"])
        self.fc3 = nn.Linear(parameters["hidden_size1"], parameters["hidden_size2"])

        self.fc4 = nn.Linear(parameters["hidden_size1"], parameters["num_classes"])
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        out = self.fc1(x)
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
    

def compute_accuracy(model, eval_dataloader):
    """
    Compute model performance/accuracy on eval dataloader set.

    model: trained pytorch nn
    eval_dataloader: validation huggingface dataloader

    """

    cuda =  torch.cuda.is_available()
    device = torch.device("cuda") if cuda else torch.device("cpu")

    accuracies = []
    model.eval()
    length = len(eval_dataloader)
    for eval_batch in eval_dataloader:
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        
        outputs = model(eval_batch["word_feature"])
        labels = eval_batch["label"]
        for i,output in enumerate(outputs):
            index = max_(output)
            actual = labels[i].item()
            if index == actual: accuracies.append(1)
            else: accuracies.append(0)

    accuracy = sum(accuracies)/len(accuracies)
    print("Validation data accuracy is {}".format(accuracy))

def input_fun(sentence, word_model):
    """
    This function takes as input a list of sentences and converts them into a list
    of features.
    sentence: list of tagged tokens
    word_model: trained fasttext embedding of dimension 50
    
    returns:
    Y: list of labels(tags lying between 0 to 11)
    X: list of feature vectors (900 dimensional)
    
    """
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
        
        try:
            vector_curr = word_model.wv[X_sentence[i]]
        except:
            sim = word_model.wv.most_similar(positive=[X_sentence[i]])[0][0]
            vector_curr = word_model.wv[sim]
        
        if i != 0:
            try:
                vector_prev = word_model.wv[X_sentence[i-1]]
            except:
                sim = word_model.wv.most_similar(positive=[X_sentence[i-1]])[0][0]
                vector_prev = word_model.wv[sim]
        else:
            vector_prev = np.zeros(300)
        
        if i != 0 and i != 1:
            try:
                vector_prevv = word_model.wv[X_sentence[i-2]]
            except:
                sim = word_model.wv.most_similar(positive=[X_sentence[i-2]])[0][0]
                vector_prevv = word_model.wv[sim]
        else:
            vector_prevv = np.zeros(300)
        
        X_triplet.extend(vector_curr)
        X_triplet.extend(vector_prev)
        X_triplet.extend(vector_prevv)
        
        
        X = X + [X_triplet]
        Y = Y + [Y_sentence[i]]
    
    return Y, X

def evaluate(model, criterion, valid_data, word_model, tag_index, device):
    """
    This function evaluates the neural network on the validation dataloader

    model: trained model
    eval_dataloader: Pytorch dataloader for evaluating the trained model
    device: device on which to run the model(cpu/gpu)

    returns:
    avg_loss: loss over the validation data

    """
    
    losses = []
    model.eval()
    
    batches = 0
    for eval_sent in valid_data:
        if batches == 1000: break
        Y, X = input_fun(eval_sent, word_model)
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
    checkpoint_path = parameters["OUT_DIR"] + "NN_POS_Tagging/best_checkpoint/"
    print("Total steps are {}".format(len(train_data)*parameters["num_epochs"]))

    for epochs in range(parameters["num_epochs"]):
        for sent in train_data:
            # pass this tagged sentence for training
            net.train()
            Y, X = input_fun(sent, word_model)
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
                    torch.save(net.state_dict(), checkpoint_path + "nn_model.pt")
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

    return net
