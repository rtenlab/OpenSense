#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import MultipleEVM
import EVM
import h5py
import torch
import time
from IPython.utils import io
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
try:
    import cPickle
except:
    import _pickle as cPickle
    
def nptorch(data):
    return torch.from_numpy(np.array(data))

distance_function='cosine'
device = torch.device("cuda")
delta = 0.35
tailsize=100
cover_threshold=0.7
distance_multiplier=0.4#0000000000001
nb_classes = 6
sudo_label = np.arange(nb_classes)
nb_cl = 3
nb_groups = 6


#LOAD EVM features
EVM_features_datasets = 'DNN_EVM_features_PAMAP2.pickle'
with open(EVM_features_datasets,'rb') as fp:
    X_sep_features = cPickle.load(fp)
    X_sep_label = cPickle.load(fp)
    X_sep_features_val = cPickle.load(fp)
    X_sep_label_val = cPickle.load(fp)
    X_sep_features_test = cPickle.load(fp)
    X_sep_label_test = cPickle.load(fp)
    
train_data = []
for i in range(18):
    train_data.append([])
print(train_data)
for i in range(18):    
    for j in range(len(X_sep_features)):
        if X_sep_label[j] == i:
            train_data[i].append(X_sep_features[j])
    print(len(train_data[i]))

train_data = np.array(train_data)
mevem_update = MultipleEVM.MultipleEVM(tailsize=tailsize, cover_threshold=cover_threshold, distance_multiplier=distance_multiplier, distance_function=distance_function, device=device)
mevem_update.train([nptorch(train_data[0]),nptorch(train_data[1]),nptorch(train_data[2]),nptorch(train_data[3]),nptorch(train_data[4]),nptorch(train_data[5]),
                   nptorch(train_data[6]),nptorch(train_data[7]),nptorch(train_data[8]),nptorch(train_data[9]),nptorch(train_data[10]),
                   nptorch(train_data[11]),nptorch(train_data[12]),nptorch(train_data[13]),nptorch(train_data[14]),nptorch(train_data[15]),
                   nptorch(train_data[16]),nptorch(train_data[17])], 
                   labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
probabilities = mevem_update.probabilities(torch.FloatTensor(X_sep_features_val))
class_prob = mevem_update.class_probabilities(probabilities = probabilities)
pred_class_prob = np.argmax(class_prob,axis=1) #+ 1
true_labels = np.asarray(X_sep_label_val)#.argmax(axis=1)
accuracy = accuracy_score(true_labels, pred_class_prob)
macro = f1_score(true_labels, pred_class_prob, average='macro')
micro = f1_score(true_labels, pred_class_prob, average='micro')

print('accuracy', accuracy)
print('macro', macro)
print('micro', micro)

probabilities = mevem_update.probabilities(torch.FloatTensor(X_sep_features_test))
class_prob = mevem_update.class_probabilities(probabilities = probabilities)
pred_class_prob = np.argmax(class_prob,axis=1) #+ 1
true_labels = np.asarray(X_sep_label_test)#.argmax(axis=1)
accuracy = accuracy_score(true_labels, pred_class_prob)
macro = f1_score(true_labels, pred_class_prob, average='macro')
micro = f1_score(true_labels, pred_class_prob, average='micro')

print('testaccuracy', accuracy)
print('tetstmacro', macro)
print('testmicro', micro)

cm = confusion_matrix(true_labels, pred_class_prob)
print(cm)
print("Precision: ",precision_score(true_labels, pred_class_prob, average='macro'))
print("Recall: ",recall_score(true_labels, pred_class_prob,  average='macro'))
print("F1: ",f1_score(true_labels, pred_class_prob, average="macro"))


# In[ ]:




