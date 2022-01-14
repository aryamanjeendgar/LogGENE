import numpy as np
import torch
# import torchvision.datasets as datasets
import logging
import os
from os import path
from sklearn.model_selection import KFold
import pandas as pd
import zipfile
import urllib.request


class UCIDatasets():
    def __init__(self,  name,  data_path="", n_splits = 10):
        self.datasets = {
            "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
            }
        self.data_path = data_path
        self.name = name
        self.n_splits = n_splits
        self._load_dataset()

    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name) # making a request for the file and storing it under UCI/, only downloads the file that it was asked to  
        data = None # instantiating the variable for storing the data

        # simple ladder for choosing which file to load
        if self.name == "housing":
            data = pd.read_csv(self.data_path+'UCI/housing.data',
                        header=0, delimiter="\s+").values # parsing the data
            self.data = data[np.random.permutation(np.arange(len(data)))] # randomly permuting the data

        elif self.name == "concrete":
            data = pd.read_excel(self.data_path+'UCI/Concrete_Data.xls',
                               header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]


        elif self.name == "energy":
            data = pd.read_excel(self.data_path+'UCI/ENB2012_data.xlsx',
                                 header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]


        elif self.name == "power":
            zipfile.ZipFile(self.data_path +"UCI/CCPP.zip").extractall(self.data_path +"UCI/CCPP/")
            data = pd.read_excel(self.data_path+'UCI/CCPP/Folds5x2_pp.xlsx', header=0).values
            np.random.shuffle(data)
            self.data = data


        elif self.name == "wine":
            data = pd.read_csv(self.data_path + 'UCI/winequality-red.csv',
                               header=1, delimiter=';').values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + 'UCI/yacht_hydrodynamics.data',
                               header=1, delimiter='\s+').values
            self.data = data[np.random.permutation(np.arange(len(data)))]
            

        kf = KFold(n_splits=self.n_splits) # instantiating the KFold split object
        self.in_dim = data.shape[1] - 2
        self.out_dim = 2
        self.data_splits = kf.split(data)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]


    def get_split(self, split=-1, train=True):
        """
        processes the data in standard ways (normalizing et. al) and finally returns the respective torch tensors -- already need to have instantiated dataset
        """
        if split == -1:
            split = 0
        if 0<=split and split<self.n_splits: 
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index,
                                    :self.in_dim], self.data[train_index, self.in_dim:]
            x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
            y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0)**0.5
            x_train = (x_train - x_means)/x_stds
            y_train = (y_train - y_means)/y_stds
            x_test = (x_test - x_means)/x_stds
            y_test = (y_test - y_means)/y_stds
            if train:
                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).float()
                train_data = torch.utils.data.TensorDataset(inps, tgts)
                return train_data
            else:
                inps = torch.from_numpy(x_test).float()
                tgts = torch.from_numpy(y_test).float()
                test_data= torch.utils.data.TensorDataset(inps, tgts)
                return test_data
