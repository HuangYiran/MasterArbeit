#-*- coding: UTF-8 -*-
import sys
sys.path.append('utils/')
sys.path.append('models/')
import torch
import argparse
import numpy as np
import random
import math

from torch.optim import lr_scheduler
#from torch.utils.data import DataLoader
from ELMoMetric import *
from nnLoss import *

parser = argparse.ArgumentParser()
#onmt.Markdown.add_md_help_argument(parser)
# model
parser.add_argument('-model', required = True, default = '../data/mt_model/model_deen', help = 'Path to model.pt file')
# data for training and testing
parser.add_argument('-src', required = True, help = 'source sentence file')
parser.add_argument('-tgt_s1', required = True, help = 'target sentence file if necessary')
parser.add_argument('-tgt_s2', required = True, help = 'target sentence of system two')
parser.add_argument('-tgt_ref', required = True, help = 'reference target sentence')
parser.add_argument('-scores', required = True, help = 'scores data for training')
parser.add_argument('-test_src', required = True, help = 'source sentence for testing')
parser.add_argument('-test_tgt_s1', required = True, help = 'target sentence of system one for testing')
parser.add_argument('-test_tgt_s2', required = True, help = 'target sentence of system two for resting')
parser.add_argument('-test_tgt_ref', required = True, help = 'target sentence of reference for testing')
parser.add_argument('-test_scores', required = True, help = 'scores data for testing')
# path to save the output
parser.add_argument('-output', default = '/tmp/', help = 'path to save the output')
# others
parser.add_argument('-gpu', type = int, default = -1, help = "device to run on")
parser.add_argument('-batch_size', type = int, default = 30, help = 'batch size')
parser.add_argument('-max_sent_length', type = int, default = 100, help = 'maximum sentence length')
parser.add_argument('-replace_unk', action='store_true', help = """...""")
parser.add_argument('-verbose', action = 'store_true', help = 'Print scores and predictions for each sentence')
parser.add_argument('-cuda', default = False)
def main():
    opt = parser.parse_args()
    # set models and loss
    model = ELMoMetric(opt)
    loss = torch.nn.NLLLoss()
    # set optimizer
    optimizer = torch.optim.Adam(model.downStream.parameters(), lr = 1e-3)
    # set lr scheduler
    lamb1 = lambda x: .1**(x//30)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lamb1)
    # read data
    train = Data(opt.src, opt.tgt_s1, opt.tgt_s2, opt.tgt_ref, opt.scores)
    test = Data(opt.test_src, opt.test_tgt_s1, opt.test_tgt_s2, opt.test_tgt_ref, opt.test_scores)
    dl_train = DataLoader(train, batch_size = opt.batch_size, shuffle = True)
    dl_test = DataLoader(test, batch_size = opt.batch_size, shuffle = True)
    # train the model 
    num_epochs = 10
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        for batch_idx, dat in enumerate(dl_train()):
            src = dat[0]
            tgt_s1 = dat[1]
            tgt_s2 = dat[2]
            tgt_ref = dat[3]
            scores = dat[4]
            optimizer.zero_grad()
            out = model(src, tgt_s1, tgt_s2, tgt_ref)
            lo = loss(out, scores)
            lo.backward()
            train_loss += lo.data[0]
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(src),
                    len(dl_train.dataset), 100.*batch_idx/len(dl_train),
                    lo.data[0]/len(src)
                    ))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch,
                train_loss/len(dl_train.dataset)
                ))
        test_loss = test(dl_test, model, loss)
        print('====> Epoch: {} Average test loss: {:.4f}'.format(
                epoch,
                test_loss/len(dl_test.dataset)
                ))
        
    torch.save(model, opt.save_dir+'/'+opt.model)

def test(dl_test, model, loss):
    model.eval()
    test_loss = 0
    for batch_idx, dat in enumerate(dl_test()):
        src = dat[0]
        tgt_s1 = dat[1]
        tgt_s2 = dat[2]
        tgt_ref = dat[3]
        scores = dat[4]
        out = model(src, tgt_s1, tgt_s2, tgt_ref)
        lo = loss(out, scores)
        test_loss += lo.data[0]
    return test_loss

#class Data(torch.utils.data.Dataset):
class Data:
    def __init__(self, src, tgt_s1, tgt_s2, tgt_ref, scores):
        """ 
        这个方法被我用崩了，原来的作用一点都没有体现出来，
        it may be beter to use array list??? """
        #super(Data, self).__init__()
        self.data = {}
        self.data['src'] = self.add_file(src)
        self.data['tgt_s1'] = self.add_file(tgt_s1)
        self.data['tgt_s2'] = self.add_file(tgt_s2)
        self.data['tgt_ref'] = self.add_file(tgt_ref)
        self.data['scores'] = self.add_scores(scores)
        assert(len(self.data['src']) == len(self.data['tgt_s1']) and 
               len(self.data['src']) == len(self.data['tgt_s2']) and
               len(self.data['src']) == len(self.data['tgt_ref']) and
               len(self.data['src']) == len(self.data['scores']))
        self.len = len(self.data['src'])
    
    def add_file(self, path):
        return [li.rstrip('\n').split(' ') for li in open(path)]
    
    def add_scores(self, path):
        return [int(li.rstrip('\n')) for li in open(path)]
    
    def get_data(self):
        return self.data
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['src'][index], 
                self.data['tgt_s1'][index], 
                self.data['tgt_s2'][index], 
                self.data['tgt_ref'][index],
                self.data['scores'][index])

class DataLoader:
    def __init__(self, data, batch_size, shuffle):
        self.data = data.get_data()
        self.batch_size = batch_size
        self.length = len(data)
        self.num_batch = int(math.ceil(self.length*1./self.batch_size))
        if shuffle:
            self._shuffle()
    
    def __call__(self):
        for index in range(self.num_batch):
            src = []
            s1 = []
            s2 = []
            ref = []
            scores = []
            for i in range(index*self.batch_size, (index + 1)* self.batch_size):
                if i >= self.length:
                    break
                src.append(self.data['src'][i])
                s1.append(self.data['tgt_s1'][i])
                s2.append(self.data['tgt_s2'][i])
                ref.append(self.data['tgt_ref'][i])
                scores.append(self.data['scores'][i])
            yield((src, s1, s2, ref, scores))
    
    def _shuffle(self):
        order = range(self.length)
        random.seed('23423')
        random.shuffle(order)
        self.data['src'] = [self.data['src'][i] for i in order]
        self.data['tgt_s1'] = [self.data['tgt_s1'][i] for i in order]
        self.data['tgt_s2'] = [self.data['tgt_s2'][i] for i in order]
        self.data['tgt_ref'] = [self.data['tgt_ref'][i] for i in order]
        self.data['scores'] = [self.data['scores'][i] for i in order]
    
if __name__ == '__main__':
    main()

