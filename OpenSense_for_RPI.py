#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import MultipleEVM
import EVM
import h5py
import torch
import time
from IPython.utils import io
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
#import seaborn as sn
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.special import softmax
try:
    import cPickle
except:
    import _pickle as cPickle
    
    
import os

import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from collections import OrderedDict
from finch import FINCH
import cv2
from B3 import calc_b3

from MultipleEVM import MultipleEVM
from torch.cuda.amp import autocast 

#import sys
#orig_stdout = sys.stdout
#f = open('opensense_owm.txt', 'w')
#sys.stdout = f

distance_function='cosine'
device = torch.device("cuda") #CHANGE TO "cpu" for RPI
delta = 0.35
tailsize=100
cover_threshold=0.7
distance_multiplier=0.4
nb_classes = 18
sudo_label = np.arange(nb_classes)
init_nb_cl = 9
nb_cl = 3
nb_groups = 4

#LOAD EVM features
EVM_features_datasets = str(nb_cl)+'EVM_features_PAMAP2_init.pickle'
with open(EVM_features_datasets,'rb') as fp:
    X_sep_features = cPickle.load(fp)
    X_sep_label = cPickle.load(fp)
    X_sep_features_val = cPickle.load(fp)
    X_sep_label_val = cPickle.load(fp)
    X_sep_features_test = cPickle.load(fp)
    X_sep_label_test = cPickle.load(fp)
    
with open(str(nb_cl)+'PAMAP2_mixing_init.pickle','rb') as fp:
    mixing = cPickle.load(fp)
    
known_list = mixing[0:9]
unknown_list = mixing[9:18]
print(known_list,unknown_list)
groups = [[9, 10, 11],[12, 13, 14],[15, 16, 17]]
test_featrues = []
test_labels = []
for i in known_list:
    test_featrues.extend(X_sep_features_test[i])
    test_labels.extend(X_sep_label_test[i])

print( np.unique(test_labels))
for iteration in range(3):
    owl_train_labels = []
    owl_train_featrues = []

    for i in (groups[iteration]):
        owl_train_featrues.extend(X_sep_features[i])
        owl_train_labels.extend(X_sep_label[i])
        owl_train_featrues.extend(X_sep_features_val[i])
        owl_train_labels.extend(X_sep_label_val[i])
        test_featrues.extend(X_sep_features_test[i])
        test_labels.extend(X_sep_label_test[i])
    #print(len(owl_train_featrues))
    #print(len(test_featrues))
    #print( np.unique(owl_train_labels))
    #print( np.unique(test_labels))
    
def compute_B3(y,preds_cat):
    L = y#true_labels
    C = preds_cat#pred_class_prob

    is_known = (L>=0) * (L<9)
    is_unknown = ~is_known
    predicted_known = (C>=0) * (C<9)
    predicted_unknown = ~predicted_known

    #print(is_known,is_unknown)

    N_KK =  np.sum(is_known*predicted_known)
    N_KU =  np.sum(is_known*predicted_unknown)
    N_UK =  np.sum(is_unknown*predicted_known)
    N_UU =  np.sum(is_unknown*predicted_unknown)

    N_ALL = N_KK + N_KU + N_UK + N_UU


    LKK = L[is_known*predicted_known]
    CKK = C[is_known*predicted_known]
    LUU = L[is_unknown*predicted_unknown]
    CUU = C[is_unknown*predicted_unknown]

    if N_KK > 0:
      correct = np.sum(LKK==CKK)
    else:
      correct = 0
    if N_UU > 0:
      b3, _, _ = calc_b3(L = LUU , K = CUU)
    else:
      b3 = 0 
    OWM = ( correct +  ( b3 * N_UU ) ) /  N_ALL

    print("OWM = ",OWM)

    


class OpenSense(object):
  def __init__(self,
         csv_folder, cores, detection_threshold):    


    self.csv_folder = csv_folder
    self.cores = cores
    self.detection_threshold = detection_threshold

    self.T = detection_threshold
    self.UU = 0
    
    self.queue_dict = {} #empty dictionary
    self.clustered_set= set() #empty set
    self.clustered_dict= {}
    
    evm_known_feature_path = evm_model_path

    self.rho = number_of_unknown_to_create_evm
    self.psi = number_of_unknown_to_strat_clustering
    self.number_known_classes = N_known_classes 
    

    # initialize EVM


    self.evm = MultipleEVM(tailsize=tailsize,
                 cover_threshold=cover_threshold,
                 distance_multiplier=distance_multiplier)
    self.evm.load(evm_model_path)
    

  def test_B3(self, features, labels):
    FVs = torch.from_numpy(features)
    Pr = self.evm.class_probabilities(FVs)
    pred_class_prob = np.argmax(Pr,axis=1) #+ 1
    true_labels = np.asarray(labels)#.argmax(axis=1)
    #accuracy = accuracy_score(true_labels, pred_class_prob)
    #macro = f1_score(true_labels, pred_class_prob, average='macro')
    #print('testaccuracy', accuracy)
    #print('tetstmacro', macro)
    #cm = confusion_matrix(true_labels, pred_class_prob)
    #print(cm)
    #size = max(len(np.unique(true_labels)),len(np.unique(pred_class_prob)))
    #print(size)
    #df_cm = pd.DataFrame(cm, range(size), range(size))
    #normed_c = (df_cm.T / df_cm.astype(np.float).sum(axis=1)).T
    #plt.figure(figsize=(16,12))
    #sn.set(font_scale=1.4) # for label size
    #sn.heatmap(normed_c, annot=True, annot_kws={"size": 16},fmt='.2%', cmap='Blues') # font size
    #plt.show()
    compute_B3(true_labels,pred_class_prob)
    
  def feature_extraction(self, test_features, test_labels):
    #load features
    len_ = test_features.shape[0]
    self.features_dict = {}
    self.label_dict = {}

    for index, feature, label in zip(range(len_),test_features,test_labels):
        self.features_dict[index] = test_features[index]
        self.label_dict[index] = test_labels[index]
    
    return self.features_dict
    
  def ow_classification(self, round_id):

    
    
    result_path = os.path.join(self.csv_folder, 
                  f"class_" + str(round_id).zfill(2)+".csv")
    image_names, FVs = zip(*self.features_dict.items()) 
    FVs = np.asarray(FVs, dtype=np.float32)
    FVs = torch.from_numpy(FVs)

    Pr = self.evm.class_probabilities(FVs)
    Pr = torch.tensor(Pr)
    Pm,_ = torch.max(Pr, dim=1)
    pu = 1 - Pm
    #print('SHAPES',pu.shape,Pr.shape)
    all_rows_tensor = torch.cat((pu.view(-1,1), Pr), 1)
    #print('SHAPES',all_rows_tensor.shape)

    norm = torch.norm(all_rows_tensor, p=1, dim=1)
    normalized_tensor = all_rows_tensor/norm[:,None]
    col1 = ['id', 'P_unknown']
    col2 = ['P_'+str(k) for k in range(1, self.number_known_classes+1)]
    col3 = ['U_'+str(k) for k in range(1, self.UU+1)]
    col = col1 + col2 + col3


    self.df_classification = pd.DataFrame(zip(image_names,*normalized_tensor.t().tolist()), columns=col)
    self.df_classification.to_csv(result_path, index = False, header = False, float_format='%.4f')
    
    result_path_raw = os.path.join(self.csv_folder, 
                  f"raw_class_" + str(round_id).zfill(2)+".csv")
    self.df_class_raw = pd.DataFrame(zip(image_names,*all_rows_tensor.t().tolist()), columns=col)
    self.df_class_raw.to_csv(result_path_raw, index = False, header = False, float_format='%.4f')
    return result_path,self.df_classification


  def model_updating(self, features, df_classification, round_id=None):
    """
    Update evm models

    """
    
    m = -2
    nu = 0 
    for k, row in df_classification.iterrows():
        if row[1] > self.T:  # predicted unknown unknown #before > greater
            self.queue_dict[k] = features[k] 
        if len(self.queue_dict) >= self.psi:
            data =  np.vstack(self.queue_dict.values())
            c_all, num_clust, req_c = FINCH(data, verbose=True)
            cluster_labels = c_all[:,-1]
            m = num_clust[-1]  # number of clusters after clustering. 
            to_be_delete = []
        if m >= 2:
            FVsn_queue, FVs_queue = zip(*self.queue_dict.items())
            if len(self.clustered_dict)>0:
                image_names_clustered, FVs_clustered = zip(*self.clustered_dict.items())
            else:
                FVs_clustered=[]
                for k in range(m):  # number of clusters after clustering. 
                    index = [i for i in range(len(cluster_labels)) if cluster_labels[i] == k]
                    index_neg = [i for i in range(len(cluster_labels)) if cluster_labels[i] != k]
                    if len(index) > self.rho:
                        to_be_delete = to_be_delete + index
                        nu = nu+1
                        FV_positive = torch.from_numpy(np.array([FVs_queue[k] for k in index]))
                        FV_negative_1 = [FVs_queue[k] for k in index_neg]
                        FV_negative_2 = list(FVs_clustered)
                        FV_negative = torch.from_numpy(np.array(FV_negative_1 + FV_negative_2))
                        y = self.number_known_classes + self.UU + nu
                        # Train a new EVM with FV_positive and [FV_negative_1+FVs_clustered] as negatives
                        # Insert the new EVM to new_EVM_list
                        self.evm.train_update(new_points = FV_positive, label = (y-1), distance_multiplier = unknown_dm , extra_negatives = FV_negative )



            
    if nu > 0:
        fv_covered = []
        for k in (to_be_delete):
            fv_covered.append(FVsn_queue[k])
        for name in fv_covered:
            fv_name = self.queue_dict[name]
            self.clustered_dict.update({name:fv_name})
        del self.queue_dict[name]
    self.UU = self.UU + nu
    #print("End: len(self.clustered_dict) = ", len(self.clustered_dict))
    #print("End: len(self.queue_dict) = ", len(self.queue_dict))
    #print(f"{nu} new evm classes added. Total discovered classes = {self.UU}")
    return self.evm



  def save(self, name):
    self.evm.save(f'/scratch/OpenSense_EVM_{name}.hdf5')

####################################


csv_folder = './csv_folder/PAMAP2_9/'
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

number_of_tests = 0

N_CPU = 32 #CHANGE to 4 for RPI
batch_size = 100 #set to 10 for RPI
start_learning = 500

evm_model_path = './data/EVM_PAMAP2_9.hdf5'

feature_size = 64



tailsize = 2000
cover_threshold = 0.7
distance_multiplier = 0.4
unknown_dm = 0.6 #distance multiplier for unknown classes

N_known_classes = 9
number_of_unknown_to_create_evm = 70
number_of_unknown_to_strat_clustering = 400

cores = 32
detection_threshold = 0.01#0.001 #delta

t0 = time.time()





csv_folder_i = csv_folder
OpenSense_alg = OpenSense(csv_folder_i, cores, detection_threshold)

t1 = time.time()
print(f"Loading time {t1-t0}")
start_time = time.time()


groups = [[9, 10, 11],[12, 13, 14],[15, 16, 17]]
test_featrues = []
test_labels = []
for i in known_list:

    test_featrues.extend(X_sep_features_test[i])
    test_labels.extend(X_sep_label_test[i])

print( np.unique(test_labels))
for iteration in range(3):
    owl_train_labels = []
    owl_train_featrues = []

    for i in (groups[iteration]):
        owl_train_featrues.extend(X_sep_features[i])
        owl_train_labels.extend(X_sep_label[i])
        owl_train_featrues.extend(X_sep_features_val[i])
        owl_train_labels.extend(X_sep_label_val[i])
        test_featrues.extend(X_sep_features_test[i])
        test_labels.extend(X_sep_label_test[i])
    print(len(owl_train_featrues))
    print(len(test_featrues))
    print( np.unique(owl_train_labels))
    print( np.unique(test_labels))
    LDSF_test_all = torch.from_numpy(np.asarray(test_featrues))
    owl_test_data = np.asarray(test_featrues)
    owl_test_labels = np.asarray(test_labels)
    ds_test_features = np.asarray(owl_train_featrues)
    ds_test_labels = np.asarray(owl_train_labels)

    print(ds_test_features.shape,ds_test_labels.shape)
    features_list = ds_test_features
    labels_list = ds_test_labels
    num_rounds = (len(features_list)) //batch_size

    if ( (len(features_list)) % batch_size) !=0 :
        num_rounds += 1

    print("num_rounds = ", num_rounds) ## num_rounds in this case, basically number of batches
    df_classification = pd.DataFrame()
    features = {}
    o_len = 0
    for round_id in range(num_rounds):
        t2 = time.time()

        feature_batch = features_list[round_id*batch_size : (round_id+1)*batch_size]
        label_batch = labels_list[round_id*batch_size : (round_id+1)*batch_size]

        t3 = time.time()
        F = OpenSense_alg.feature_extraction(feature_batch, label_batch)
        t4 = time.time()
        _, C = OpenSense_alg.ow_classification(round_id)
        t5 = time.time()

        c_len = len(F)
        idx = np.arange(0,c_len)

        for index in idx:
            features[index+o_len] = F[index]
        o_len =o_len + c_len
        df_classification = pd.concat([df_classification, C])
        if o_len+1 >= start_learning:
            df_classification = df_classification.reset_index(drop=True)
            #print("f len,c len,olen,clen : ", len(features),len(df_classification),o_len,c_len)
            evm = OpenSense_alg.model_updating(features,df_classification, round_id)
            t6 = time.time()
            #print("model_updating time = ", t6-t5)
            df_classification = pd.DataFrame()
            features = {}
            o_len = 0
        #os.remove(round_file_name)
    if o_len > 0:
        evm = OpenSense_alg.model_updating(features,df_classification, round_id)
        t6 = time.time()
        #print("model_updating time = ", t6-t5)
    t7 = time.time()

    #print("round time = ", t7-t2)
    OpenSense_alg.test_B3(owl_test_data,owl_test_labels)

del OpenSense_alg
end_time = time.time()
#print(f"Loading time {t1-t0}")





#sys.stdout = orig_stdout
#f.close()







# In[ ]:




