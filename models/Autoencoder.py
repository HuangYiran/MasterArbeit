#-*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
sys.path.append('../utils/')
import nnActi

class VAE(nn.Module):
    def __init__(self, dim2 = 256, dim3 = 64):
        dim = 500
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(dim, dim2)
        self.fc21 = nn.Linear(dim2, dim3)
        self.fc22 = nn.Linear(dim2, dim3)
        self.fc3 = nn.Linear(dim3, dim2)
        self.fc4 = nn.Linear(dim2, dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, x):
        h3 = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

class Autoencoder(nn.Module):
    def __init__(self, dim2 = 256, dim3 = 64, act_func = 'Tanh'):
        """
        最后一层要不要激活函数呢，还有到底要几层呢？？
        """
        super(Autoencoder, self).__init__()
        dim = 500
        self.encoder = nn.Sequential(
                torch.nn.Linear(dim, dim2),
                nnActi.get_acti(act_func),
                torch.nn.Linear(dim2, dim3),
                )
        self.decoder = nn.Sequential(
                torch.nn.Linear(dim3, dim2),
                nnActi.get_acti(act_func),
                torch.nn.Linear(dim2, dim),
                )
    def forward(self, data):
        tmp = self.encoder(data)
        out = self.decoder(tmp)
        return out
