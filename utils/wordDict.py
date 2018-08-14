#-*- coding: utf-8 -*-
from tqdm import tqdm
import torch

class WordDict:
    def __init__(self, path, add_signs = True, resume = True, save_dicts = False):
        """
        the data used here should already be cleaned,
        path: string or list of string
            path of the dataset or the sentence list 
        """
        if resume:
            print('-- loading dicts from the checkpoint  wordDict')
            checkpoint = torch.load('./checkpoints/wordDict')
            self.sents = checkpoint['sents']
            self.dict = checkpoint['num_dict']
            self.freq_dict = checkpoint['freq_dict']
            self.num_of_words = len(self.dict)
            print('-- finish')
        else:
            print('-- init word dict')
            self.dict = {}
            if type(path) == str:
                self.sents = [li.rstrip('\n') for li in open(path)]
            else:
                self.sents = path
            print('num of sentences: %d'%(len(self.sents)))
            self.dict = self._get_num() # based on self.sents
            if add_signs:
                self.dict['BOS'] = len(self.sents)
                self.dict['EOS'] = len(self.sents)
            self.num_of_words = len(self.dict)
            print('num of words: %d'%(self.num_of_words))
            self.freq_dict = self._get_freq()
            print('-- finish')
        if save_dicts:
            self.save_dicts()

    def get_freq_dict(self):
        return self.freq_dict

    def get_freq_of_word(self, word):
        if word not in self.freq_dict:
            return 0
        else:
            return self.freq_dict[word]
    
    def get_freq_of_words(self, words):
        """
        words is list of word
        """
        out = []
        for word in words:
            out.append(self.get_freq_of_word(word))
        return out

    def get_num_dict(self):
        return self.dict

    def get_num_of_word(self, word):
        if word not in self.dict:
            return 0
        else:
            return self.dict[word]

    def save_dicts(self):
        print('-- saving the dicts')
        torch.save({
            'sents': self.sents,
            'num_dict': self.dict,
            'freq_dict': self.freq_dict
            }, './checkpoints/wordDict')
        print('-- finish')

    def _get_freq(self):
        """
        here we instead of div len(words), we div number of sentence ?????
        """
        tmp = self.dict.copy()
        for key in self.dict.keys():
            tmp[key] = tmp[key]*1./len(self.sents)
        return tmp

    def _get_num(self):
        tmp = self.dict.copy()
        for sent in tqdm(self.sents):
            words = sent.split(' ')
            for word in words:
                if word in tmp.keys():
                    tmp[word] += 1
                else:
                    tmp[word] = 1
        return tmp
