#!/usr/bin/env python
import math
import warnings
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch.utils.data as data_utils
import matplotlib.pyplot as plt



class classificationNetwork(nn.Module):
    def __init__(self, indim, size1, size2):
        super(classificationNetwork,self).__init__()
        self.l1 = nn.Linear(indim, size1)
        self.l2 = nn.Linear(size2,10)
        self.l3 = nn.Linear(10,1)

    def forward(self,x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.sigmoid(self.l3(x))
        return x

    # Used in LALR
    def penU(self, x):
        op = F.tanh(self.l1(x))
        op = F.tanh(self.l2(op))
        return op


class regressionNetwork(nn.Module):
    def __init__(self, size1, size2, indim, outdim, drop):
        super(regressionNetwork, self).__init__()
        self.l1= nn.Linear(indim, size1)
        self.l2= nn.Dropout(p= drop)
        self.l3= nn.Linear(size1, size2)
        self.l4= nn.Dropout(p= drop)
        self.l5= nn.Linear(size2, outdim)

    def forward(self, x):
        x= F.tanh(self.l1(x))
        x= F.tanh(self.l3(self.l2(x)))
        x= self.l5(self.l4(x))
        return x

    def penU(self, x):
        x= F.tanh(self.l2(self.l1(x)))
        x= F.tanh(self.l4(self.l3(x)))
        return x
