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
# data for testing
parser.add_argument('-tgt_s1', required = True, help = 'target sentence file if necessary')
parser.add_argument('-tgt_s2', required = True, help = 'target sentence of system two')
parser.add_argument('-tgt_ref', required = True, help = 'reference target sentence')
parser.add_argument('-scores', required= True, help = 'the target')
# output file
parser.add_argument('-output', default = '/tmp/decState_params', help = 'path to save the output')
# others
parser = argparse.ArgumentParser()
parser.add_argument('-model', default = 'autoencoder')
parser.add_argument('-checkpoint', default = './checkpoints/autoencoder')

def main():
    opt = parser.parse_args()
    # load model
    model = torch.load(opt.checkpoint)
    # load data
    test = Data(opt.tgt_s1, opt.tgt_s2, opt.tgt_ref, opt.scores)
    dl_test = DataLoader(test, batch_size = opt.batch_size, shuffle = True)
    # set loss
    loss = torch.nn.NLLoss()
    # cal loss
    test_loss, test_taul = test_model(dl_test, model, loss)
    # print result
    print('====> Average loss: {:.4f}\tAverage Taul: {:.4f}'.format(
                train_loss/len(dl_train),
                train_taul/len(dl_train),
                ))
    
def test_model(dl_test, model, loss):
    model.eval()
    test_loss = 0
    test_taul = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_test()):
        #if counter == 20:
        #    break;
        counter += 1
        src = dat[0]
        tgt_s1 = dat[1]
        tgt_s2 = dat[2]
        tgt_ref = dat[3]
        scores = dat[4]
        out = model(src, tgt_s1, tgt_s2, tgt_ref)
        lo = loss(out, scores)
        test_loss += lo.data[0]
        taul = evaluate_tau_like(out, scores)
        test_taul += taul
    return test_loss/counter, test_taul/counter

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


