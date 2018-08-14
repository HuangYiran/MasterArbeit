#-*- coding: utf-8 -*-
"""
get sentencee embedding from word embeding
"""
import sys
sys.path.append('./utils')
import numpy as np
import sklearn
import torch

from wordDict import WordDict
from tqdm import tqdm
from sklearn.decomposition import PCA

class WR:
    def __init__(self, path):
        """
        path: string or list of string 
            the path of the dataset that used to create the dict or the list of sentence
        """
        # load dict
        self.dict = WordDict(path)
        self.num_dict= self.dict.get_num_dict()
        self.freq_dict = self.dict.get_freq_dict()
        self.alpha = 1e-3

    def forward(self, sents, wordEs):
        """
        !!!false now, please copy the code from forward2 first
        only work for nn hidden layer data, it has the following specialities:
            1. start with a start sign
            2. fix seq length, append with 0(???may be not)
            3. all sentence length smaller as fix seq length
        sents: string or list of string 
            path of the file set or list of sentence
        wordEs: string or numpy.matrix
            path of the dataset, the data should be 
            numpy matrix (batch_size, seq_len, num_dim) 
            or is the data
        """
        seq_len = 100
        # cal Vs
        if type(wordEs) == str:
            wordEs= np.load(wordEs)
        else:
            wordEs = wordEs
        if type(sents) == str:
            sents = [li.rstrip('\n').split(' ') for li in open(sents)] # load sentence
        else:
            sents = sents
        assert(len(wordEs) == len(sents))
        len_sents = torch.FloatTensor([len(li) for li in sents]) # get length of each sentense (batch_size)
        sents = [li.insert(0, 'BOS') for li in sents] # add BOS sign
        sents = [li.append('EOS') for li in sents] # add EOS sign
        freq_sents = torch.FloatTensor([map(lambda x: self.dict.get_freq_of_words(x), sent) for sent in sents]) # get the freq
        freq_sents.append(torch.FloatTensor(seq_len)) # should be the max length
        padded_freq_sents = torch.nn.utils.rnn.pad_sequence(freq_sents, batch_fiost = True)[:-1].unsqueeze(1) # (batch_size, 1, seq_len)
        pref = self.alpha/(self.alpha + padded_freq_sents)
        unit = len_sents * torch.bmm(pref, wordEs) # (batch_size, num_dim)
        # pca
        unit = unit.numpy()
        pca = PCA(unit.shape[1], svd_solver = 'auto')
        pca.fit(unit)
        out = unit - np.dot(np.dot(unit, pca.components_.T), pca.components_)
        return out
    
    def forward2(self, sents_li, wordEs_li):
        """
        compare with forward, this function works when more then one dataset are given
        doesn't accept the path
        only work for nn hidden layer data, it has the following specialities:
            1. start with a start sign
            2. fix seq length, append with 0(???may be not)
            3. all sentence length smaller as fix seq length
        sents: string or list of string 
            path of the file set or list of sentence
        wordEs: string or numpy.matrix
            path of the dataset, the data should be 
            numpy matrix (batch_size, seq_len, num_dim) 
            or is the data
        """
        seq_len = 100
        assert(len(sents_li) == len(wordEs_li))
        unit = []
        for index in tqdm(range(len(sents_li))):
            if type(sents_li[index]) == str:
                sents = [li.rstrip('\n').split(' ') for li in open(sents_li[index])]
            else:
                sents = sents_li[index]
                sents = [li.split(' ') for li in sents]
            if type(wordEs_li[index]) == str:
                wordEs= torch.from_numpy(np.load(wordEs_li[index]))
            else:
                wordEs = wordEs_li
            assert(len(sents) == len(wordEs))
            len_sents = torch.FloatTensor([len(li) for li in sents]) # get length of each sentense (batch_size)
            sents = [self._insert(li, 'BOS') for li in sents] # add BOS sign
            sents = [self._append(li, 'EOS') for li in sents] # add EOS sign
            freq_sents = [map(lambda x: self.dict.get_freq_of_word(x), sent) for sent in sents] # get the freq
            freq_sents = [torch.FloatTensor(li) for li in freq_sents]
            print "*"*10
            print freq_sents[:10]
            #freq_sents.append(torch.FloatTensor(seq_len)) # should be the max length
            padded_freq_sents = self._pad_sequence(freq_sents).unsqueeze(1) # (batch_size, 1, seq_len)
            pref = self.alpha/(self.alpha + padded_freq_sents)
            bmm = torch.bmm(pref, wordEs).squeeze()
            for index in range(bmm.shape[0]):
                bmm[index] = bmm[index] * (1.0/len_sents[index])
            unit.append(bmm)
        unit = torch.cat(unit, 0) # (batch_size, num_dim)
        # pca
        type(unit)
        unit = unit
        unit = unit.numpy()
        pca = PCA(unit.shape[1], svd_solver = 'auto')
        pca.fit(unit)
        #out = unit - np.dot(np.dot(unit, pca.components_.T), pca.components_)
        out = pca.transform(unit)
        #out = unit
        print out.shape
        return out.astype(np.float32)

    def _append(self, li, string):
        tmp = li
        tmp.append(string)
        return tmp

    def _insert(self, li, string):
        tmp = li
        tmp.insert(0, string)
        return tmp

    def _pad_sequence(self, data, max_length = 100):
        PAD = 10
        max_length = max_length
        out = data[0].new(len(data), max_length).fill_(PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        return out

