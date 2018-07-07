# -*- coding: UTF-8 -*-
import sys
sys.path.append("./utils/")
sys.path.append("./models/")
import torch
import argparse
import numpy as np
import random
import math
import os
import nnInit
import nnLoss
import nnPlot

from torch.utils.data import DataLoader
from scipy import stats
from data import DataUtil
from LinearModel import BasicLinear, BasicLinear_dropout, BiLinear, TwoLayerLinear
from MaskedModel import MaskedModel1, MaskedModel2, MaskedModel3
from FullHiddenModel import *
from RankModel import *
from Params import Params
from hyperopt import fmin, tpe, hp
from valuation import valTauLike
from transform import daToRr
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
# data for training
parser.add_argument('-tgt_s1', required = True, help = 'target sentence file if necessary')
parser.add_argument('-tgt_s2', required = True, help = 'target sentence of system two')
parser.add_argument('-tgt_ref', required = True, help = 'reference target sentence')
parser.add_argument('-scores', required= True, help = 'the target')
# output file
parser.add_argument('-output', default = '/tmp/decState_params', help = 'path to save the output')
# others
parser.add_argument('-seq_len', type = int, default = 40, help = 'set the max length of the sequence')
parser.add_argument('-batch_size', type = int, default = 100, help = 'batch size')
parser.add_argument('-combine_data', default = False, help = 'combine the data before input to the model')


def main():
    opt = parser.parse_args()
    # set models and loss
    model = ELMo()
    loss = torch.nn.NLLLoss()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # set lr scheduler
    lamb1 = lambda x: .1**(x//30)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lamb1)
    # read data
    train = Data(opt.tgt_s1, opt.tgt_s2, opt.tgt_ref, opt.scores)
    dl_train = DataLoader(train, batch_size = opt.batch_size, shuffle = True)
    # train the model 
    num_epochs = 100
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        train_taul = 0
        for batch_idx, dat in enumerate(dl_train):
            tgt_s1 = torch.autograd.Variable(dat[0], requires_grad = False)
            tgt_s2 = torch.autograd.Variable(dat[1], requires_grad = False)
            tgt_ref = torch.autograd.Variable(dat[2], requires_grad = False)
            scores = torch.autograd.Variable(dat[3], requires_grad = False)
            if opt.combine_data:
                inp = torch.cat([tgt_s1, tgt_s2, tgt_ref], 1)
            optimizer.zero_grad()
            if opt.combine_data:
                out = model(inp)
            out = model(tgt_s1, tgt_s2, tgt_ref)
            lo = loss(out, scores)
            lo.backward()
            train_loss += lo.data[0]
            optimizer.step()
            taul = evaluate_tau_like(out, scores)
            train_taul += taul
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTaul: {:.6f}'.format(
                    epoch,
                    batch_idx * opt.batch_size,
                    len(train), 1.*opt.batch_size*batch_idx/len(train),
                    lo.data[0],
                    taul
                    ))
        print('====> Epoch: {} Average loss: {:.4f}\tAverage Taul: {:.4f}'.format(
                epoch,
                train_loss/len(dl_train),
                train_taul/len(dl_train),
                ))
        test_loss, test_taul = test_model(dl_test, model, loss)
        print('====> Epoch: {} Average test loss: {:.4f}\tAverage test taul: {:.4f}'.format(
                epoch,
                test_loss,
                test_taul
                ))
        
    torch.save(model, opt.output+'/'+opt.model)



def result_transform_sf_to_score(x):
    a, b, c = x[0], x[1], x[2]
    if a > b and a > c:
        return -1
    elif b > a and b > c:
        return 0
    elif c > a and c > b:
        return 1
    else:
        # ???
        return 0

def evaluate_tau_like(arr1, arr2):
    """
    arr1 comes from the model
    arr2 comes from the target file
    """
    a1 = arr1.cpu()
    a2 = arr2.cpu()
    a1 = a1.data.numpy()
    a2 = a2.data.numpy()
    a1 = list(map(result_transform_sf_to_score, a1))
    a2 = a2 - 1
    taul = valTauLike(a2, a1) # a2 should go first
    return taul

class Data:
    def __init__(self, tgt_s1, tgt_s2, tgt_ref, scores):
        """ 
        这个方法被我用崩了，原来的作用一点都没有体现出来，
        it may be beter to use array list??? 
        """
        #super(Data, self).__init__()
        self.data = {}
        self.data['tgt_s1'] = self.add_file(tgt_s1)
        self.data['tgt_s2'] = self.add_file(tgt_s2)
        self.data['tgt_ref'] = self.add_file(tgt_ref)
        self.data['scores'] = self.add_scores(scores)
        assert(len(self.data['scores']) == len(self.data['tgt_s1']) and 
               len(self.data['scores']) == len(self.data['tgt_s2']) and
               len(self.data['scores']) == len(self.data['tgt_ref']))
        self.len = len(self.data['scores'])
    
    def add_file(self, path):
        """
        return Variable of torch.FloatTensor
        """
        return torch.from_numpy(np.load(path))
    
    def add_scores(self, path):
        # for softmax output, so we add one for each score
        return torch.LongTensor([int(li.rstrip('\n')) + 1 for li in open(path)])
    
    def get_data(self):
        return self.data
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['tgt_s1'][index], 
                self.data['tgt_s2'][index], 
                self.data['tgt_ref'][index],
                self.data['scores'][index])

if __name__ == '__main__':
    main()

