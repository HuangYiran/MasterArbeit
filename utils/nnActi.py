import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Parameter(torch.Tensor([1]))
    
    def forward(self, input):
        acti  = input * self.sigmoid(self.beta * input)
        return acti

def get_acti(act_fun):
    #print '***********************'
    #print globals()
    return getattr(nn, act_fun)() if hasattr(nn, act_fun) else globals()[act_fun]()