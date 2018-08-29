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

from scipy.stats import stats
from distance import canberra
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
#from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, mahalanobis, minkowski, seuclidean, sqeuclidean, wminkowski

parser = argparse.ArgumentParser()
parser.add_argument('-tgt_s1', required = True, help = 'target sentence file if necessary')
parser.add_argument('-tgt_ref', required = True, help = 'reference target sentence')
parser.add_argument('-scores', required= True, help = 'the target')
parser.add_argument('-test_s1', required = True, help = 'test s1')
parser.add_argument('-test_ref', required = True, help = 'test ref')
parser.add_argument('-test_scores', required = True, help = 'test scores')
# output file
parser.add_argument('-output', default = '/tmp/decState_params', help = 'path to save the output')
# others
parser.add_argument('-model', default = 'distance', help = 'set the type of the model including: [distance, mepper]')
parser.add_argument('-seq_len', type = int, default = 40, help = 'set the max length of the sequence')
parser.add_argument('-batch_size', type = int, default = 30, help = 'batch size')
parser.add_argument('-num_dim', type = int, default = 300, help = 'number of dimension of input data')
parser.add_argument('-combine_data', default = False, help = 'combine the data before input to the model')
parser.add_argument('-resume', default = False, help = 'set true, to load a existed model and continue training')
parser.add_argument('-checkpoint', help = 'only work when resume setted true, point to the address of the model') # only for model: ELMo_modified
parser.add_argument('-cand', nargs = '+', type = int, help = 'list of int, store the code of the features that will be used in the model')
parser.add_argument('-verbose', action='store_true')


###################
# main function
###################

def main():
    print '*'*100
    opt = parser.parse_args()
    # set models and loss
    model = Simple(opt)
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # set scheduler
    lamb1 = lambda x: .1**(x//30)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lamb1)
    # load data
    train = Data(opt.tgt_s1, opt.tgt_ref, opt.scores, opt.cand)
    test = Data(opt.test_s1, opt.test_ref, opt.test_scores, opt.cand)
    dl_train = DataLoader(train, batch_size = opt.batch_size, shuffle = True)
    dl_test = DataLoader(test, batch_size = opt.batch_size, shuffle = True)
    # train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        counter = 0
        train_corr = 0
        for batch_idx, dat in enumerate(dl_train):
            counter += 1
            # devide the data and transform them to torch.Variable
            tgt_s1 = torch.autograd.Variable(dat[0], requires_grad = False)
            tgt_ref = torch.autograd.Variable(dat[1], requires_grad = False)
            scores = torch.autograd.Variable(dat[2], requires_grad = False)
            #scores = scores.float() # for L1Loss
            # run the model
            if opt.combine_data:
                inp = torch.cat([tgt_s1, tgt_ref], 1)
            optimizer.zero_grad()
            if opt.combine_data:
                out = model(inp)
            out = model(tgt_s1, tgt_ref)
            # cal the loss and update the params
            lo = loss(out, scores)
            lo.backward()
            optimizer.step()
            train_loss += lo.data
            corr = evaluate_corr(out, scores)
            train_corr += corr
            if opt.verbose:
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tpearson corr: {:.6f}'.format(
                        epoch,
                        batch_idx * opt.batch_size,
                        len(train),
                        100.*batch_idx*opt.batch_size/len(train),
                        lo.data,
                        corr
                        ))
        test_loss, test_corr = test_model(dl_test, model, loss)
        if opt.verbose:
            # train
            print('====> Epoch: {} Average train loss: {:.4f}\tpearson corr: {:.4f}'.format(
                epoch,
                train_loss/counter,
                100 * train_corr/counter
                ))
            # test
            print('====> Epoch: {} Average test loss: {:.4f}\tpearson corr: {:.4f}'.format(
                epoch,
                test_loss,
                100 * test_corr
                ))
        else:
            if epoch == num_epochs - 1:
                print('train pearson: {:.6f}\ttest pearson: {:.6f}'.format(100* train_corr/counter, 100*test_corr))

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
    a1 = arr1.cpu()
    a2 = arr2.cpu()
    a1 = a1.data.numpy()
    a2 = a2.data.numpy()
    #a1 = list(map(result_transform_sf_to_score, a1))
    #a1 = a1 - 1 # for L1Loss
    #a2 = a2 - 1
    pearsonr = stats.pearsonr(a1, a2)[0]
    return pearsonr

def test_model(dl_test, model, loss):
    model.eval()
    test_loss = 0
    test_corr = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_test):
        #if counter == 20:
        #    break;
        counter += 1
        tgt_s1 = torch.autograd.Variable(dat[0], requires_grad = False)
        tgt_ref = torch.autograd.Variable(dat[1], requires_grad = False)
        scores = torch.autograd.Variable(dat[2], requires_grad = False)
        out = model(tgt_s1, tgt_ref)
        lo = loss(out, scores.float()) #for margin loss
        test_loss += lo.data
        corr = evaluate_corr(out, scores)
        test_corr += corr
    return test_loss/counter, test_corr/counter


######################
## model
######################
class Simple(torch.nn.Module):
    """
    more layers input distance base 1: with nn
    讨论一下，是什么原因造成的testset的结果比trainingset的差那么多，如果是同一年的数据，是否这种差距会缩小，如果是的话，是否可以通过一部分数据的预训练来提升结果。
    """
    def __init__(self, opt):
        """
        当li输出为1维时效果很差，说明一个维度并不足以表达足够的信息。。
        """
        super(Simple, self).__init__()
        dim1 = opt.num_dim
        dim2 = 500
        if opt.cand:
            self.num_layers = len(opt.cand)
        else:
            self.num_layers = 0
        act_func = 'ReLU'
        self.num_dim = dim1 
        self.weight_layers = torch.nn.Parameter(torch.ones(self.num_layers), requires_grad = True)
        self.weight_dimension = torch.nn.Parameter(torch.randn(self.num_dim), requires_grad = True)
        self.sf = torch.nn.Softmax(1)
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(dim1, dim2))
        
        self.mlp2 = torch.nn.Sequential()
        self.mlp2.add_module('fc1', torch.nn.Linear(dim1, dim2))
 

    def forward(self, s1, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        # expand the weight parameter
        if self.num_layers != 0 and self.num_layers != 1:
            batch_size, num_layers, num_dim = s1.data.shape
            assert(self.num_layers == num_layers and self.num_dim == num_dim)
            ewl = self.weight_layers.expand(batch_size, self.num_layers) # ==> (batch_size, num_layer)
            ewl = self.sf(ewl).unsqueeze(1) # ==> (batch_size, 1, num_layer)
            # process the data with ewl
            s1 = torch.bmm(ewl, s1).squeeze() # ==> (batch_size, num_dim)
            ref = torch.bmm(ewl, ref).squeeze()
        # process the data with ewd
        s1 = self.mlp(s1)
        ref = self.mlp(ref)
        # compute the distance 
        #d1 = canberra(s1, ref)
        d1 = torch.nn.functional.pairwise_distance(s1, ref, p = 1)
        return -d1

if __name__ == '__main__':
    main()
