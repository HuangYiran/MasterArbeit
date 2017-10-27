# -*- coding: UTF-8 -*-
import sys
sys.path.append("./utils/")
sys.path.append("./models/")
import torch
import argparse
import numpy
import random

from data import DataUtil
from LinearModel import BasicLinear

parser = argparse.ArgumentParser(description = "run.py")

parser.add_argument('-model', default = './model/LinearModel', 
        help = 'path to model')
parser.add_argument('-src_sys', default = './data/hidden_sys',
        help = 'path to hidden value of system out')
parser.add_argument('-src_ref', default = './data/hidden_ref',
        help = 'path to thidden value of references')
parser.add_argument('-tgt', default = './data/data_scores',
        help = 'path to the score file')
parser.add_argument('-out', default = './test_data/pred',
        help = 'path to save the model output')
parser.add_argument('-batch_size', default = 50,
        help = 'the size of each batch')
parser.add_argument('-lr', default = 0.02,
        help = 'learning rate')
parser.add_argument('-eps', default = 1e-08)
parser.add_argument('-weight_decay', default = 0)

opt = parser.parse_args()

def main():
    # read data
    data = DataUtil(opt)

    # build model
    model = BasicLinear()

    # set optimizer and loss functioin
    # lr = 0.002, betas = (0.9, 0.888), eps = 1e-08, weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr, eps = opt.eps, weight_decay = opt.weight_decay)
    loss_fn = torch.nn.MSELoss()

    # get the number of batch 
    nu_batch = data.get_nu_batch()
    print("number of batch is %d"%(nu_batch))

    # train
    for i in range(5 * nu_batch):
        src, tgt = data.get_batch_repeatly()
        train_batch(model, loss_fn, optimizer, src, tgt)
        if i % 10 == 0:
            print("evaluate %d" %(i/10))
            evaluate(model, src, tgt)

def train_batch(model, loss_fn, optimizer, src, tgt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)

    optimizer.zero_grad()
    out = model(src)
    loss = loss_fn(out, tgt)
    loss.backward()
    optimizer.step()
    return True

def evaluate(model, src, tgt):
    arr1 = predict(model, src)
    arr2 = tgt.view(1, -1)
    arr = torch.cat((arr1, arr2), 0).numpy()
    corr = numpy.corrcoef(arr)[0][1]
    print("the correlation coeffizient is : %f" %(corr))
    return True

def predict(model, src):
    src = torch.autograd.Variable(src, requires_grad = False)
    out = model(src)
    return out.view(1, -1).data

if __name__ == "__main__":
    main()
