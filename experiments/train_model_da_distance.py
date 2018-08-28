# -*- coding: UTF-8 -*-
"""
for the mepper model
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
import scipy.stats as stats

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, mahalanobis, minkowski, seuclidean, sqeuclidean, wminkowski

parser = argparse.ArgumentParser()
parser.add_argument('-tgt_s1', required = True, help = 'target sentence file if necessary')
parser.add_argument('-tgt_ref', required = True, help = 'reference target sentence')
parser.add_argument('-scores', required= True, help = 'the target')
# output file
parser.add_argument('-output', default = '/tmp/decState_params', help = 'path to save the output')
# others
parser.add_argument('-type', default = 'L1', help = 'set the type of distance method')
parser.add_argument('-batch_size', type = int, default = 100, help = 'batch_size')
parser.add_argument('-cand', nargs = '+', type = int, help = 'list of int, store the code of the features that will be used in the model')
parser.add_argument('-verbose', action='store_true')


###################
# main function
###################
def main():
    opt = parser.parse_args()
    # load data
    train = Data(opt.tgt_s1, opt.tgt_ref, opt.scores, opt.cand)
    dl_train = DataLoader(train, batch_size = opt.batch_size, shuffle = True)
    # calculate distance
    corr = test_model(dl_train, model, opt)
    print '#'*100
    print opt.type
    print corr


 
##################
# assist function 
##################
class Data:
    def __init__(self, tgt_s1, tgt_ref, scores, cand):
        """ 
        这个方法被我用崩了，原来的作用一点都没有体现出来，
        it may be beter to use array list??? 
        """
        #super(Data, self).__init__()
        self.cand = cand
        self.data = {}
        self.data['tgt_s1'] = self.add_file(tgt_s1)
        self.data['tgt_ref'] = self.add_file(tgt_ref)
        self.data['scores'] = self.add_scores(scores)
        assert(len(self.data['scores']) == len(self.data['tgt_s1']) and 
               len(self.data['scores']) == len(self.data['tgt_ref']))
        self.len = len(self.data['scores'])
    
    def add_file(self, path):
        """
        return Variable of torch.FloatTensor
        """
        if self.cand:
            out = torch.from_numpy(np.load(path))[:,self.cand, :]
            if len(self.cand) == 1:
                out = out.squeeze(1)
            return out
        else:
            return torch.from_numpy(np.load(path))
    
    def add_scores(self, path):
        # for softmax output, so we add one for each score
        return torch.FloatTensor([float(li.rstrip('\n')) for li in open(path)])
    
    def get_data(self):
        return self.data
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['tgt_s1'][index], 
                self.data['tgt_ref'][index],
                self.data['scores'][index])

def evaluate_corr(arr1, arr2):
    """
    arr1 comes from the model
    arr2 comes from the target file
    """
    #print(type(arr1))
    #print(arr1)
    #print(type(arr2))
    #print(arr2)
    a1 = arr1
    a2 = arr2.data.numpy()
    #a1 = list(map(result_transform_sf_to_score, a1))
    #a1 = a1 - 1 # for L1Loss
    #a2 = a2 - 1
    pearsonr = stats.pearsonr(a1,a2)[0]
    return pearsonr

def test_model(dl_test, model, opt):
    test_corr = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_test):
        counter += 1
        tgt_s1 = torch.autograd.Variable(dat[0], requires_grad = False)
        tgt_ref = torch.autograd.Variable(dat[1], requires_grad = False)
        scores = torch.autograd.Variable(dat[2], requires_grad = False)
        out = model(tgt_s1, tgt_ref, opt)
        corr = evaluate_corr(out, scores)
        test_corr += corr
    return test_corr/counter

######################
## model
######################
def L1(v1, v2):
    #return np.absolute(v1-v2).sum()
    return np.linalg.norm(v1-v2, ord = 1)

def L2(v1, v2):
    return np.linalg.norm(v1-v2)

def cos(v1, v2):

    mul = np.dot(v1,v2)
    n1  = np.linalg.norm(v1)
    n2 = np. linalg.norm(v2)
    return mul*1./(n1*n2)

def mulsum(v1,v2):
    return sum(v1*v2)

def model(rf, hf, opt):
    assert(len(rf) == len(hf))
    distances = []
    if opt.type == 'mahalanobis':
        rh = torch.cat([rf,hf], 0)
        ic = np.linalg.inv(np.cov(rh, rowvar = False))
    if opt.type == 'seuclidean':
        rh = torch.cat([rf,hf])
        v = np.var(rh, 0)
    for k in range(len(rf)):
        i = rf[k]
        j = hf[k]
        if opt.type == 'L1':
            distance = L1(i,j)
        elif opt.type == 'L2':
            distance = L2(i,j)
        elif opt.type == 'cos':
            distance = cos(i,j)
        elif opt.type == 'braycurtis':
            distance = braycurtis(i,j)
        elif opt.type == 'canberra':
            distance = canberra(i, j)
        elif opt.type == 'chebyshev':
            distance = chebyshev(i,j)
        elif opt.type == 'cityblock':
            distance = cityblock(i,j)
        elif opt.type == 'correlation':
            distance = correlation(i,j)
        elif opt.type == 'mahalanobis':
            distance = mahalanobis(i,j,ic)
        elif opt.type == 'minkowski':
            distance = minkowski(i,j, 3)
        elif opt.type == 'mulsum':
            distance = mulsum(i, j)
        elif opt.type == 'seuclidean':
            distance = seuclidean(i,j,v)
        elif opt.type == 'sqeuclidean':
            distance = sqeuclidean(i,j)
        distances.append(distance)
    return distances

if __name__ == '__main__':
    main()
