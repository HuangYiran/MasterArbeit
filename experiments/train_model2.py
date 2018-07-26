# -*- coding: UTF-8 -*-
"""
train_model is for ranking data
train_model is not
"""
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
from LinearModel import BasicLinear, BasicLinear_dropout, BiLinear, TwoLayerLinear, Simple1, Simple2, Simple3, Simple4, Simple5, Simple6, Simple7, Simple8
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
parser.add_argument('-test_s1', required = True, help = 'test s1')
parser.add_argument('-test_s2', required = True, help = 'test s2')
parser.add_argument('-test_ref', required = True, help = 'test ref')
parser.add_argument('-test_scores', required = True, help = 'test scores')
# output file
parser.add_argument('-output', default = '/tmp/decState_params', help = 'path to save the output')
# others
parser.add_argument('-seq_len', type = int, default = 40, help = 'set the max length of the sequence')
parser.add_argument('-batch_size', type = int, default = 100, help = 'batch size')
parser.add_argument('-combine_data', default = False, help = 'combine the data before input to the model')
parser.add_argument('-resume', default = False, help = 'set true, to load a existed model and continue training')
parser.add_argument('-checkpoint', help = 'only work when resume setted true, point to the address of the model') # only for model: ELMo_modified
parser.add_argument('-cand', nargs = '+', type = int, help = 'list of int, store the code of the features that will be used in the model')
parser.add_argument('-verbose', action='store_true')


def main():
    opt = parser.parse_args()
    # set models and loss
    if opt.resume:
        model = torch.load(opt.checkpoint)
    model = Simple8(len(opt.cand))
    #loss = torch.nn.MSELoss()
    #loss = torch.nn.SoftMarginLoss() #0.35/0.43
    #loss = torch.nn.HingeEmbeddingLoss() #.34
    #loss = torch.nn.SmoothL1Loss() #-.35/0.44
    loss = torch.nn.L1Loss() # for L1Loss
    #loss = nnLoss.CorrLoss()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # set lr scheduler
    lamb1 = lambda x: .1**(x//30)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lamb1)
    # read data
    train = Data(opt.tgt_s1, opt.tgt_s2, opt.tgt_ref, opt.scores, opt.cand)
    test = Data(opt.test_s1, opt.test_s2, opt.test_ref, opt.test_scores, opt.cand)
    dl_train = DataLoader(train, batch_size = opt.batch_size, shuffle = True)
    dl_test = DataLoader(test, batch_size = opt.batch_size, shuffle = True)
    # train the model 
    num_epochs = 10
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        train_taul = 0
        counter = 0
        for batch_idx, dat in enumerate(dl_train):
            counter += 1
            tgt_s1 = torch.autograd.Variable(dat[0], requires_grad = False)
            tgt_s2 = torch.autograd.Variable(dat[1], requires_grad = False)
            tgt_ref = torch.autograd.Variable(dat[2], requires_grad = False)
            scores = torch.autograd.Variable(dat[3], requires_grad = False)
            #scores = scores.float() # for L1Loss
            if opt.combine_data:
                inp = torch.cat([tgt_s1, tgt_s2, tgt_ref], 1)
            optimizer.zero_grad()
            if opt.combine_data:
                out = model(inp)
            out = model(tgt_s1, tgt_s2, tgt_ref)
            target = torch.autograd.Variable(torch.ones(out.data.shape)*10, requires_grad = False)
            #out1 = out*scores.float()
            #lo = loss(out1, target)
            lo = loss(out,scores.float()) # for margin loss
            lo.backward()
            train_loss += lo.data
            optimizer.step()
            taul = evaluate_tau_like(out, scores)
            train_taul += taul
            if opt.verbose:
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTaul: {:.6f}'.format(
                        epoch,
                        batch_idx * opt.batch_size,
                        len(train), 100.*opt.batch_size*batch_idx/len(train),
                        lo.data,
                        taul
                        ))
        if opt.verbose:
            print('====> Epoch: {} Average train loss: {:.4f}\tAverage Taul: {:.4f}'.format(
                    epoch,
                    train_loss/counter,
                    train_taul/counter,
                    ))
        # cal loss
        test_loss, test_taul = test_model(dl_test, model, loss)
        # print result
        if opt.verbose:
            print('====> Epoch: {} Average test loss: {:.4f}\tAverage Taul: {:.4f}'.format(
                        epoch,
                        test_loss,
                        test_taul,
                        ))
        else:
            if epoch == num_epochs-1:
                print 'Average test loss: ' + str(test_taul)
        torch.save(model, opt.output)

def test_model(dl_test, model, loss):
    model.eval()
    test_loss = 0
    test_taul = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_test):
        #if counter == 20:
        #    break;
        counter += 1
        tgt_s1 = torch.autograd.Variable(dat[0], requires_grad = False)
        tgt_s2 = torch.autograd.Variable(dat[1], requires_grad = False)
        tgt_ref = torch.autograd.Variable(dat[2], requires_grad = False)
        scores = torch.autograd.Variable(dat[3], requires_grad = False)
        out = model(tgt_s1, tgt_s2, tgt_ref)
        target = torch.autograd.Variable(torch.ones(out.data.shape)*10, requires_grad = False)
        #out1 = out*scores.float()
        #lo = loss(out1, target)
        lo = loss(out, scores.float()) #for margin loss
        test_loss += lo.data
        taul = evaluate_tau_like(out, scores)
        test_taul += taul
    return test_loss/counter, test_taul/counter



def result_transform_sf_to_score(x):
    if x>0:
        return 1
    elif x==0:
        return 0
    else:
        return -1

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
    #a1 = a1 - 1 # for L1Loss
    #a2 = a2 - 1
    taul = valTauLike(a2, a1) # a2 should go first
    return taul

class Data:
    def __init__(self, tgt_s1, tgt_s2, tgt_ref, scores, cand):
        """ 
        这个方法被我用崩了，原来的作用一点都没有体现出来，
        it may be beter to use array list??? 
        """
        #super(Data, self).__init__()
        self.cand = cand
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
        if self.cand:
            return torch.from_numpy(np.load(path))[:,self.cand, :]
        else:
            return torch.from_numpy(np.load(path))
    
    def add_scores(self, path):
        # for softmax output, so we add one for each score
        return torch.LongTensor([int(li.rstrip('\n')) for li in open(path)])
    
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

