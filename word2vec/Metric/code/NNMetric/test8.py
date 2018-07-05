#-*- coding: utf-8 -*-
"""
used in script emb-nn.py, to get the sentence embedding from word embedding
the method implemented here is wr, becaue i need all the data(include train, val, test data) to train the pca
Therefore, this method is not implemented in file test4
"""
import sys
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/models')
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/MasterArbeit/utils')
import argparse
import torch
import numpy as np

from tqdm import tqdm
from sentEmbd import WR

parser = argparse.ArgumentParser(description = 'test.py')
parser.add_argument('-ref', help = 'Reference file')
parser.add_argument('-s1', help = 'data from system 1')
parser.add_argument('-s2', help = 'data from system 2')
parser.add_argument('-test_ref', help = 'reference test data')
parser.add_argument('-test_s1', help = 'test data from system 1')
parser.add_argument('-test_s2', help = 'test data from system 2')
parser.add_argument('-sent_ref', help = 'the reference sentence set')
parser.add_argument('-sent_s1', help = 'the sentence set from system 1')
parser.add_argument('-sent_s2', help = 'teh sentence set from system 2')
parser.add_argument('-sent_test_ref', help = '...')
parser.add_argument('-sent_test_s1', help = '...')
parser.add_argument('-sent_test_s2', help = '...')
parser.add_argument('-output', default = '/tmp/', help = 'path to save the data')


def main():
    opt = parser.parse_args()
    # read sent data
    print('==> load sentence')
    sent_ref = [li.rstrip('\n') for li in open(opt.sent_ref)]
    sent_s1 = [li.rstrip('\n') for li in open(opt.sent_s1)]
    sent_s2 = [li.rstrip('\n') for li in open(opt.sent_s2)]
    sent_test_ref = [li.rstrip('\n') for li in open(opt.sent_test_ref)]
    sent_test_s1 = [li.rstrip('\n') for li in open(opt.sent_test_s1)]
    sent_test_s2 = [li.rstrip('\n') for li in open(opt.sent_test_s2)]
    # combine sent data
    sents = []
    sents.extend(sent_ref)
    sents.extend(sent_s1)
    sents.extend(sent_s2)
    sents.extend(sent_test_ref)
    sents.extend(sent_test_s1)
    sents.extend(sent_test_s2)
    # combine data in list
    sent_li = [sent_ref, sent_s1, sent_s2, sent_test_ref, sent_test_s1, sent_test_s2]
    data_li = [opt.ref, opt.s1, opt.s2, opt.test_ref, opt.test_s1, opt.test_s2]
    len_sent_li = [len(sent) for sent in sent_li]
    print sum(len_sent_li)
    cumsum_len = torch.cumsum(torch.FloatTensor(len_sent_li), 0).numpy()
    cumsum_len = map(int, cumsum_len)
    #assert(sum(len_data_li) == sum(len_sent_li))
    # init wr
    print('==> init wr')
    wr = WR(sents)
    # get sent embds
    print('==> get sent embds')
    out = wr.forward2(sent_li, data_li)
    assert(sum(len_sent_li) == len(out))
    ref = out[:cumsum_len[0]]
    s1 = out[cumsum_len[0]:cumsum_len[1]]
    s2 = out[cumsum_len[1]:cumsum_len[2]]
    test_ref = out[cumsum_len[2]: cumsum_len[3]]
    test_s1 = out[cumsum_len[3]: cumsum_len[4]]
    test_s2 = out[cumsum_len[4]: cumsum_len[5]]
    # write data
    print('==> write data')
    np.save(opt.output+'tmp1', s1)
    np.save(opt.output+'tmp2', s2)
    np.save(opt.output+'tmp3', ref)
    np.save(opt.output+'test1', test_s1)
    np.save(opt.output+'test2', test_s2)
    np.save(opt.output+'test3', test_ref)


if __name__ == '__main__':
    main()
