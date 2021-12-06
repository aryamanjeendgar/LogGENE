import torch
from torch import nn
import math

def barePDF(x, tau):
    """
    Implements the hyperbolic secant distribution
    """
    # x is a torch tensor, tau is a float
    ind= (torch.sign(x)+1)/2 # mask about the origin
    quantFactor= (1-tau)*(1-ind) + tau*ind
    return 2/math.pi*torch.cosh(x).pow_(-1)*quantFactor

def bareCDF(yhat, tau):
    """
    Implements the CDF of the HCD, which is also the required conditional distribution
    """
    # yhat is a torch tensor, tau is a float
    ind= (torch.sign(yhat)+1)/2 # mask about the origin
    quantFactor= (1-tau)*ind + tau*(1-ind)
    val= tau+4*quantFactor/math.pi*torch.atan(torch.tanh(yhat/2))
    return val

def baresBQR(y, yhat, tau):
    """
    Implements the functional form of the sBQR loss for a specified quantile
    """
    # y and yhat are torch tensors, tau is a float
    val= y*torch.log(1-bareCDF(yhat, tau))+(1-y)*torch.log(bareCDF(yhat, tau))
    return val

def sBQR(y, yhat, qs):
    """
    Returns a list of learnt quantiles for the sBQR loss
    """
    # y and yhat are torch tensors and qs is a list of floats
    # yhat would be a [batch,9*outdim] output vector, have to index it accordingly
    quantilesBQRs= []
    for idx, q in enumerate(qs):
        quantilesBQRs.append(baresBQR(y, yhat[:,idx], q))
    return quantilesBQRs

class sBQRL(nn.Module):
    """
    Torch class that implements the loss function
    """
    def __init__(self):
        super(sBQRL, self).__init__()
    
    def forward(self, y, yhat, qs, model, loader):
        return torch.mean(torch.cat(sBQR(y, yhat, qs))) + 0.0025*regularization(qs, model, loader)


device= ('cuda' if torch.cuda.is_available() else 'cpu')

"""
Implementation of the loss with the regularization term
"""
def regularization(qs, model, loader):
    """
    qs: Quantiles to be learnt
    model: Current state of the torch model being trained
    loader: The current Torch dataLoader that is being iterated over
    """
    outerSum= 0
    for inputs, labels in loader:
        inputs= inputs.to(device)
        labels= labels.to(device)
        outputs= model(inputs) # again, outputs would consist of a [batch, 9*outdim] tensor
        innerSum= 0
        for idx, q in enumerate(qs):
            if idx == len(qs):
                break
            innerSum+= torch.max(torch.zeros_like(outputs[:,idx]), outputs[:,idx]-outputs[:,idx+1])
        outerSum+= torch.sum(innerSum)
    return outerSum
