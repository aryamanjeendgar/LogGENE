#My imports
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from adahessian import Adahessian, get_params_grad
import torch.optim.lr_scheduler as lr_scheduler
import math
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# Other imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils

from sklearn.preprocessing import StandardScaler
import seaborn as sns

import warnings
import matplotlib.pyplot as plt


# Reading the data:
nRowsRead = None
# METABRIC_RNA_Mutation.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/home/aryamanj/Downloads/METABRIC_RNA_Mutation.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'METABRIC_RNA_Mutation.csv'
nRow, nCol = df1.shape


# Data processing to allow for DL application:
df2 = df1.drop(columns=['patient_id', 'cancer_type', 'cancer_type_detailed', 'cohort'])
df3 = df2[df2['her2_status_measured_by_snp6'] != 'UNDEF']
df3['er_status_measured_by_ihc'] = df3['er_status_measured_by_ihc'].apply(lambda x: 1 if "Positive" in str(x) else 0)
df3['her2_status'] = df3['her2_status'].apply(lambda x: 1 if "Positive" in str(x) else 0)
df3['inferred_menopausal_state'] = df3['inferred_menopausal_state'].apply(lambda x: 1 if "Post" in str(x) else 0)
df3['primary_tumor_laterality'] = df3['primary_tumor_laterality'].apply(lambda x: 1 if "Left" in str(x) else 0)
df3['pr_status'] = df3['pr_status'].apply(lambda x: 1 if "Positive" in str(x) else 0)
df3['er_status'] = df3['er_status'].apply(lambda x: 1 if "Positive" in str(x) else 0)
dummyList = ['cellularity',
             'pam50_+_claudin-low_subtype',
             'neoplasm_histologic_grade',
             #'cancer_type_detailed',
             'tumor_other_histologic_subtype',
             'integrative_cluster',
             #'gene_classifier_subtype',
             'oncotree_code',
             'her2_status_measured_by_snp6',
             '3-gene_classifier_subtype',
             'death_from_cancer'
            ]
df4 = pd.get_dummies(df3, columns=dummyList)
df4['type_of_breast_surgery'] = df4['type_of_breast_surgery'].apply(lambda x: 1 if "MASTECTOMY" in str(x) else 0)
df5 = df4.applymap(lambda x: 0 if "0" in str(x) else 1)
y = df5["death_from_cancer_Died of Disease"]
X = df5.drop(columns=["overall_survival", "death_from_cancer_Died of Other Causes", "death_from_cancer_Living", "death_from_cancer_Died of Disease"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


batch_is= 128
# creating torch dataLoaders:
train_dataset = data_utils.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(y_train.to_numpy()))
test_dataset = data_utils.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(y_test.to_numpy()))
trainLoader= data_utils.DataLoader(train_dataset, batch_size =batch_is, pin_memory=True,shuffle=True,num_workers = 1)
testLoader= data_utils.DataLoader(test_dataset,batch_size =batch_is,pin_memory=True,shuffle = False,num_workers = 1)



# Sample use-case: create_xy(dataset, x_cols, y_col, separator, 0.4, ditch_head= remove_head)
# Code for parsing the .txt files:
def parseY(dataset, target_columns, delim, split_ratio):
    with open(dataset, 'r') as f:
        lines = f.readlines()
    X = []
    Y = []
    for line in lines:
        while len(line) > 0 and line[-1] == "\n":
            line = line[:len(line)-1]
        split_array = line.split(delim)
        all_columns = []
        for value in split_array:
            if value !="" and value !=" ":
                all_columns.append(value)
        if len(all_columns)==0:
            break
        point = []
        for i in target_columns:
            point.append(float(all_columns[i]))
        Y.append(point)
        # Y.append(float(all_columns[target_column]))
    # X_arr = np.asarray(X)
    # X_unscaled = np.asarray(X)
    # Scaler.fit(X_arr)
    # X_arr = Scaler.transform(X_arr)
    Y_arr = np.asarray(Y)
    # thresh = 0
    # Y_arr_binary = np.where(Y_arr<=thresh,0,1)
    # unique, counts = np.unique(Y_arr_binary, return_counts=True)
    # x_train, x_test, y_train, y_test = train_test_split(X_arr, Y_arr_binary, test_size = split_ratio)
    # return x_train, x_test, y_train, y_test, Y_arr, X_arr, X_unscaled
    return Y_arr
