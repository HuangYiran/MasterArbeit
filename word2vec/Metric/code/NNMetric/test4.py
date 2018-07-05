"""
Used in files emb-nn.py, to get the sentence embedding from word embedding
the joint type include:
    - mean
    - sum
    = sumN
    - sumWithGap
    - head
    - max
    - ScaledAttnAdd
    - AttenThresAdd
"""
import argparse
import gensim
import string
import re
import os
import sys
import torch
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/models')
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/utils')
from Attention import *
from sentEmbd import WR

import numpy as np
import nltk
#nltk.download('all')
from nltk.tokenize.moses import MosesTokenizer
"""
didn't calculate the distance
only use join to compare the vector if necessary
"""
parser = argparse.ArgumentParser(description='test.py')


parser.add_argument('-ref', help='Reference file')
parser.add_argument('-sent_ref', help = 'the reference sentence set')
parser.add_argument('-hyp', required=True,
                    help='Hypothesis')
parser.add_argument('-sent_hyp', help = 'the hypothesis sentence set')


parser.add_argument('-join', default="sum",
                    help='Word2VecModel')
parser.add_argument('-output', default = '/tmp/tmp1')


def distance(v1,v2):
    return np.absolute((v1 - v2)).sum()

def main():
    opt = parser.parse_args()

    data_hyp = np.load(opt.hyp)
    len_shape = len(data_hyp.shape)
    if len_shape == 2:
        out = data_hyp
    elif len_shape == 3:
        if opt.join == 'mean':
            out = data_hyp.mean(axis = 1)
        elif opt.join == 'sum':
            out = data_hyp.sum(axis = 1)
        elif opt.join == 'sumN':
            n = 30
            tmp_hyp = data_hyp[:,:n,:]
            out = tmp_hyp.sum(axis = 1)
        elif opt.join == 'sumWithGap':
            odds = torch.arange(1,20, 2).type(torch.LongTensor)
            data_hyp = torch.from_numpy(data_hyp)
            tmp_hyp = torch.index_select(data_hyp, 1, odds).numpy()
            out = tmp_hyp.sum(axis = 1)
        elif opt.join == 'head':
            out = data_hyp[:,0,:]
        elif opt.join == 'max':
            out = data_hyp.max(axis = 1)
        elif opt.join == 'AttenThresAdd':
            data_ref = np.load(opt.ref)
            out = _join_AttenThresAdd(data_hyp, data_ref)
            out = out.sum(axis = 1)
        elif opt.join == 'ScaledAttnAdd':
            data_ref = np.load(opt.ref)
            out = _join_ScaledAttnAdd(data_hyp, data_ref)
            out = out.sum(axis = 1)
        else:
            print "unreconized join type"
    else:
        print 'the shape of input data is wrong, please check the input data'
    print out.shape
    np.save(opt.output, out)

def _join_AttenThresAdd(hyp, ref):
    # turns to pytorch
    hyp = torch.from_numpy(hyp)
    ref = torch.from_numpy(ref)
    hyp = torch.autograd.Variable(hyp, requires_grad = False)
    ref = torch.autograd.Variable(ref, requires_grad = False)
    # set threshold, and softmax
    thres = torch.nn.Threshold(0.2, 0)
    softmax = torch.nn.Softmax()
    # get atten
    attn = torch.bmm(ref, hyp.transpose(1,2))
    num_batch, num_q, num_v = attn.size()
    attn = attn.view(-1, num_v)
    attn = softmax(attn)
    attn = thres(attn)
    attn = attn.view(num_batch, -1, num_v)
#    for i in range(10):
#        print(attn[i])
    out = torch.bmm(attn, hyp) # (num_batch, num_q, num_dim)
    return out.data.numpy()

def _join_ScaledAttnAdd(hyp, ref):
    hyp = torch.from_numpy(hyp)
    ref = torch.from_numpy(ref)
    hyp = torch.autograd.Variable(hyp, requires_grad = False)
    ref = torch.autograd.Variable(ref, requires_grad = False)
    # set threshold, and softmax
    softmax = torch.nn.Softmax()
    # get atten
    attn = torch.bmm(ref, hyp.transpose(1,2))/np.power(500, 0.5)
    num_batch, num_q, num_v = attn.size()
    attn = attn.view(-1, num_v)
    attn = softmax(attn)
    attn = attn.view(num_batch, -1, num_v)
    out = torch.bmm(attn, hyp) # (num_batch, num_q, num_dim)
    return out.data.numpy()



if __name__ == "__main__":
    main()
