"""
same as test5, but use cosine similarity to do the attention
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
    attn = _cal_cos_distance_pairwise(hyp, ref)
    num_batch, num_q, num_v = attn.size()
    attn = attn.view(-1, num_v)
    attn = softmax(attn)
    print(attn[0][1])
    #attn = thres(attn)
    #attn = torch.round(attn)
    attn = attn.view(num_batch, -1, num_v)
    attn = torch.max(attn, dim = 2) # (num_batch, num_v)
    out = torch.sum(attn, dim = 1)
    return out.data.numpy()

def _cal_cos_distance_pairwise(hyp, ref):
    """
    hyp and ref should be type of torch
    hyp and ref should have some shape, because torch.nn.CosineSimilarity only work with some shape
    shape of (batch_size, sen_len, num_dim)
    """
    # assert
    assert(hyp.data.shape == ref.data.shape)
    batch_size, sen_len, num_dim = ref.data.shape
    tmp = np.arange(sen_len)
    cos = torch.nn.CosineSimilarity(dim = 2)
    out = []
    for i in tqdm(range(sen_len)):
        indexes = torch.LongTensor(np.roll(tmp, -1*i))
        tmp_ref = torch.index_select(ref, 1, indexes)
        out.append(cos(hyp, tmp_ref))
    out = torch.stack(out, dim = 2)
    return out

if __name__ == "__main__":
    main()
