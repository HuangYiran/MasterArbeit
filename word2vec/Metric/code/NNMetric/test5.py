"""
Used to get the Attention distance
"""
import argparse
import gensim
import string
import re
import os
import sys
import torch
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/models')
from Attention import *

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

parser.add_argument('-hyp', required=True,
                    help='Hypothesis')


parser.add_argument('-join', default="sum",
                    help='Word2VecModel')
parser.add_argument('-output', default = '/tmp/tmp1')


def distance(v1,v2):
    return np.absolute((v1 - v2)).sum()

def main():
    opt = parser.parse_args()

    data_hyp = np.load(opt.hyp)
    data_ref = np.load(opt.ref)
    dis = _get_attn_dist(data_hyp, data_ref)
    for i in dis:
        print(i)

def _get_attn_dist(hyp, ref):
    # turns to pytorch
    hyp = torch.from_numpy(hyp)
    ref = torch.from_numpy(ref)
    hyp = torch.autograd.Variable(hyp, requires_grad = False)
    ref = torch.autograd.Variable(ref, requires_grad = False)
    # get atten
    thres = torch.nn.Threshold(0.6, 0)
    softmax = torch.nn.Softmax()
    #attn = torch.bmm(ref, hyp.transpose(1,2)) # (num_batch, num_q, num_v)
    attn = torch.bmm(hyp, ref.transpose(1,2))
    num_batch, num_q, num_v = attn.size()
    attn = attn.view(-1, num_v)
    attn = softmax(attn)
    #attn = thres(attn)
    attn = attn.view(num_batch, -1, num_v)
    print attn[1333]
    attn = torch.max(attn, dim = 1)[0] # (num_batch, num_v)
    out = torch.sum(attn, dim = 1)
    return out.data.numpy()

if __name__ == "__main__":
    main()
