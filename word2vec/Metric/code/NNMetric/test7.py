"""
use the attention to find the word paar in sys and ref sentences.
calulate the distance between the two words and sum up the distance 
we use this distance to present the difference between two sentences.
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
from tqdm import tqdm
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
    thres = torch.nn.Threshold(0.7, 0)
    softmax = torch.nn.Softmax()
    attn = _cal_cos_distance_pairwise(ref, hyp)
    num_batch, num_q, num_v = attn.size()
    attn = attn.view(-1, num_v)
    attn = softmax(attn)
    print(attn[0][1])
    #attn = thres(attn)
    #attn = torch.round(attn)
    attn = attn.view(num_batch, -1, num_v)
    attn = torch.max(attn, dim = 2)[1] # (num_batch, num_q)
    out = []
    for i in tqdm(range(num_batch)):
        dis = []
        for j in range(num_q):
            w1 = ref[i][j]
            w2 = ref[i][attn[i][j]]
            dis.append(_cal_distance(w1, w2))
        dis = torch.stack(dis, dim = 0)
        out.append(torch.sum(dis, dim = 0))
    torch.stack(out, dim = 0)
    return out.data.numpy()

def _val_distance(w1, w2, typ):
    """
    w1, w2 are type of torch
    L2
    """
    assert(w1.shape == w2.shape2)
    if typ == 'L1':
        dis = torch.norm(w1-w2, p = 1)
    elif typ == 'L2':
        dis = torch.norm(w1-w2, p = 2)
    elif typ == 'cos':
        cos = torch.nn.CosineSimilarity(dim = 0)
        dis = cos(w1, w2)
    else:
        print('unrecognized tye, set the typ to L2')
        dis = torch.norm(w1-w2, p = 2)
    return dis

def _cal_cos_distance_pairwise(hyp, ref):
    """
    hyp and ref should be type of torch
    hyp and ref should have some shape, because torch.nn.CosineSimilarity only work with some shape
    shape of (batch_size, sen_len, num_dim)
    """
    # assert
    assert(hyp.shape == ref.shape)
    batch_size, sen_len, num_dim = ref.shape
    tmp = np.arange(sen_len)
    cos = torch.nn.CosineSimilarity(dim = 2)
    out = []
    for i in range(sen_len):
        indexes = torch.LongTensor(np.roll(tmp, -1*i))
        tmp_ref = torch.index_select(ref, 1, indexes)
        out.append(cos(hyp, tmp_ref))
    out = torch.stack(out, dim = 2)
    return out

if __name__ == "__main__":
    main()
