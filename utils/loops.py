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

h= 0.4
"""
Classification training loops
"""
def trainCLRclassification(model, trainLoader, valLoader, optimizer, criterion, tau, epochs, ls_list, valList, acc_list, loss_name= "sBQC", device= "cuda"):
    """
    Training loop used for CLR training
    """
    for epoch in range(epochs):
        epoch_loss= 0.0
        # training loop
        model.train()
        for inputs, labels in trainLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            outputs= model(inputs)
            if loss_name== "BCE":
                loss= criterion(outputs.view(outputs.shape[0],), labels) # For BCE
            elif loss_name== "sBQC":
                loss= criterion(labels, outputs.view(outputs.shape[0],), tau) # For sBQC
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
        ls_list.append(epoch_loss/len(trainLoader))

        # validation loop
        val_loss= 0.0
        num_correct= 0
        total= 0
        model.eval()
        for inputs, labels in valLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            outputs= model(inputs)
            if loss_name== "BCE":
                loss= criterion(outputs.view(outputs.shape[0],), labels) # For BCE
            elif loss_name== "sBQC":
                loss= criterion(labels, outputs.view(outputs.shape[0],), tau) # For sBQC
            val_loss+= loss.item()
            x= torch.where(outputs.view(outputs.shape[0]) > 0.5, 1, 0)
            num_correct += (x==labels).sum()
            total += labels.size(0)
        valList.append(val_loss/len(valLoader))
        acc_list.append(float(num_correct)/float(total)*100)
        print("Epoch: {} Training Loss: {} Validation loss: {} Accuracy: {}".format(epoch, epoch_loss/len(trainLoader), val_loss/len(valLoader),
         float(num_correct)/float(total)*100))


def trainLRclassification(model, trainLoader, valLoader, optimizer, criterion, tau, epochs, ls_list, valList, acc_list, mask, loss_name= "sBQC", device= "cuda"):
    """
    Training loop used for LALR training
    """
    for epoch in range(epochs):
        epoch_loss= 0.0
        lr_val= classificationLR(model,train_loader, mask, tau, bSize= batch_is)
        optimizer.param_groups[0]['lr']= lr_val
        # training loop
        model.train()
        for inputs, labels in trainLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            outputs= model(inputs)
            if loss_name== "BCE":
                loss= criterion(outputs.view(outputs.shape[0],), labels) # For BCE
            elif loss_name== "sBQC":
                loss= criterion(labels, outputs.view(outputs.shape[0],), tau) # For sBQC
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
        ls_list.append(epoch_loss/len(trainLoader))

        # validation loop
        val_loss= 0.0
        num_correct= 0
        total= 0
        model.eval()
        for inputs, labels in valLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            outputs= model(inputs)
            if loss_name== "BCE":
                loss= criterion_(outputs.view(outputs.shape[0],), labels) # For BCE
            elif loss_name== "sBQC":
                loss= criterion(labels, outputs.view(outputs.shape[0],), tau) # For sBQC
            val_loss+= loss.item()
            x= torch.where(outputs.view(outputs.shape[0]) > 0.5, 1, 0)
            num_correct += (x==labels).sum()
            total += labels.size(0)
        valList.append(val_loss/len(valLoader))
        acc_list.append(float(num_correct)/float(total)*100)
        print("Epoch: {} Training Loss: {} Validation loss: {} LR: {} Accuracy: {}".format(epoch, epoch_loss/len(trainLoader), val_loss/len(valLoader), optimizer.param_groups[0]['lr'], float(num_correct)/float(total)*100))



def classificationLR(model, trainLoader, mask, tau, bSize= 16, device= "cuda"):
    """
    Takes in a network of the LALRnetwork class(during some arbitrary EPOCH of training) and the current input, and returns Kz for the EPOCH
    """
    Kz = 0.0
    model.eval()
    with torch.no_grad():
        for i,j in enumerate(trainLoader):
            inputs,labels= j[0],j[1]
            inputs= inputs.to(device)
            labels= labels.to(device)
            op1= model.penU(inputs)
            val1= torch.linalg.norm(op1)
            if val1 > Kz:
                Kz= val1

    LR= 1
    factor= 1
    if mask== 1:
        LR=  max(2/math.pi, 2-2*tau/(math.pi*tau), 2*tau/(math.pi*(1-tau)))*Kz*(1/bSize)
        factor= 0.15
    else:
        LR= 0.5*Kz*(1/bSize)
        factor= 0.05

    return (1/LR)*factor


criterion2= nn.MSELoss()
"""
Regression training loops
"""

def trainLBFGS(model, optimizer, trainLoader, valLoader, criterion, tau, epochs, ls_list, valList, loss_name, device= "cuda"):
    """
    Training loop used for LBFGS and conjugate gradient training
    """
    for epoch in range(epochs):
        epoch_loss= 0.0
        # training loop
        model.train()
        for inputs, labels in trainLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            def closure():
                optimizer.zero_grad()
                outputs= model(inputs)
                if loss_name== "MSE":
                    loss= criterion(outputs, labels)
                else:
                    loss= criterion(outputs, labels, tau, h)
                loss.backward()
                return loss
            optimizer.step(closure)
        # ls_list.append(epoch_loss/len(trainLoader))

        # validation loop
        val_loss= 0.0
        model.eval()
        for inputs, labels in valLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            outputs= model(inputs)
            loss= torch.sqrt(criterion2(outputs, labels))
            val_loss+= loss.item()
        valList.append(val_loss/len(valLoader))
        print("Epoch: {} Training loss: {} Validation loss: {}".format(epoch, epoch_loss/len(trainLoader), val_loss/len(valLoader)))

def trainCLRregression(model, optimizer, trainLoader, valLoader, criterion, tau, epochs, ls_list, valList, loss_name, device= "cuda"):
    """
    Training loop used for constantLR
    """
    for epoch in range(epochs):
        epoch_loss= 0.0
        # training loop
        model.train()
        for inputs, labels in trainLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            outputs= model(inputs)
            if loss_name== "LC":
                loss= criterion(outputs, labels, tau, h)
            else:
                loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
        ls_list.append(epoch_loss/len(trainLoader))

        # validation loop
        val_loss= 0.0
        model.eval()
        for inputs, labels in valLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            outputs= model(inputs)
            loss= torch.sqrt(criterion2(outputs, labels))
            val_loss+= loss.item()
        valList.append(val_loss/len(valLoader))
        print("Epoch: {} Training loss: {} Validation loss: {}".format(epoch, epoch_loss/len(trainLoader), val_loss/len(valLoader)))

def trainLRregression(model,optimizer, trainLoader, valLoader, criterion,  tau, epochs, ls_list, valList, loss_name, mask= False, device= "cuda"):
    """
    Training loop used for LALR training
    """
    for epoch in range(epochs):
        epoch_loss= 0.0
        lr_val= computeLRregression(model, loss_name, mask, bSize=16)
        optimizer.param_groups[0]['lr']= lr_val
        # training loop
        model.train()
        for inputs, labels in trainLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            outputs= model(inputs)
            if loss_name == "LC":
                loss= criterion(outputs, labels, tau, h)
            else:
                loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
        ls_list.append(epoch_loss/len(trainLoader))

        # validation loop
        val_loss= 0.0
        model.eval()
        for inputs, labels in valLoader:
            inputs= inputs.to(device)
            labels= labels.to(device)
            outputs= model(inputs)
            loss= torch.sqrt(criterion2(outputs, labels))
            val_loss+= loss.item()
        valList.append(val_loss/len(valLoader))
        print("Epoch: {} Training Loss: {} Validation loss: {} LR: {}".format(epoch, epoch_loss/len(trainLoader), val_loss/len(valLoader), optimizer.param_groups[0]['lr']))

# Learning rate computation functions:
def computeKa(x):
    maxNorm= 0.0
    for vector in x:
        if (maxNorm < torch.linalg.vector_norm(vector)):
            maxNorm= torch.linalg.vector_norm(vector)
    return maxNorm

def computeLRregression(model, trainLoader, ls, mask, bSize= 16, device= "cuda"):
    """
    Takes in a network of the LALRnetwork class(during some arbitrary EPOCH of training) and the current input, and returns Kz for the EPOCH
    """
    Kz = 0.0
    Ka= 0.0
    Y= 0.0
    z_k= 0.0
    model.eval()
    with torch.no_grad():
        for i,j in enumerate(trainLoader):
            inputs,labels= j[0],j[1]
            inputs= inputs.to(device)
            labels= labels.to(device)
            op1= model.penU(inputs)
            op2= model(inputs)
            val1= torch.linalg.norm(op1)
            activ2, arg2= torch.min(op2, dim= 1)
            val2, indx2= torch.min(activ2, dim= 0)
            val3= computeKa(op2)
            val4= computeKa(labels)
            if val1 > Kz:
                Kz= val1
            z_k= val2
            if val3 > Ka:
                Ka= val3
            if val3 > Y:
                Y= val4
            argMin= arg2[indx2]

    LR= 1
    factor= 1
    if ls == "LC":
        LR= (1/bSize)*Kz*torch.tanh(val4)
        factor= 0.1
    elif ls == "L1":
        LR= Kz/bSize
    elif ls == "MSE":
        LR= (1/bSize)*(Ka+Y)*Kz

    if LR==0:
        return 0.1
    if mask == 1:
        return (1/LR)*factor
    else:
        return 1/LR
