# -*- coding: UTF-8 -*-
import sys
sys.path.append("./utils/")
sys.path.append("./models/")
import torch
import argparse
import numpy
import random
import math

from data import DataUtil
from LinearModel import BasicLinear, BiLinear
from Params import Params
from hyperopt import fmin, tpe, hp

# define the objective function, this function will be used in optimization
def o_func(params):
    # set params
    opt = Params()
    opt.set_params(params)
    opt.show_params()

    # read data
    data = DataUtil(opt)

    # build model
    if opt.model == "BiLinear":
        model = BiLinear(act_func = opt.act_func, act_func_out = opt.act_func_out)
    else:
        model = BasicLinear(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate, mom = opt.momentum)
    print(model)

    # set optimizer and loss functioin
    # lr = 0.002, betas = (0.9, 0.888), eps = 1e-08, weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr, eps = opt.eps, weight_decay = opt.weight_decay)
    loss_fn = torch.nn.MSELoss()

    # get the number of batch 
    nu_batch = data.get_nu_batch()
    print("number of batch is %d"%(nu_batch))

    # train
    for i in range(50 * nu_batch):
        src, tgt = data.get_batch_repeatly()
        train_batch(model, loss_fn, optimizer, src, tgt)
        if i % 10 == 0:
            print("evaluate %d" %(i/10))
            evaluate(model, src, tgt)
    src, tgt = data.get_batch_repeatly()
    
    out = 0.0
    for i in range(10):
        src, tgt = data.get_batch_repeatly()
        out = out + evaluate(model, src, tgt)
        
    return 1 - out/10.0

# set the search space for linear Model 
li_space = hp.choice('opt',[
    {
        'type': 'linear',
        'model': './model/LinearModel',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': '../data/MasterArbeit/data/data_scores',
        'batch_size': hp.choice('li_batch_size', [50, 100, 200]),
        'lr': hp.uniform('li_lr', 0.01, 0.9),
        'dim2': hp.choice('linear_dim2', [50, 100, 500]),
        'dim3': hp.choice('linear_dim3', [None, 5, 10, 50]),
        'act_func': hp.choice('linear_act_func', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('linear_act_func_out', ['Tanh', 'Sigmoid', None]),
        'drop_out_rate': hp.uniform('drop_out_rate', 0.2, 0.8),
        'momentum': hp.uniform('bn_momentum', 0.1, 0.9),
    },
    {
        'type': 'linear',
        'model': 'BiLinear',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': '../data/MasterArbeit/data/data_scores',
        'batch_size': hp.choice('li_batch_size', [50, 100, 200]),
        'lr': hp.uniform('li_lr', 0.01, 0.9),
        'act_func': hp.choice('linear_act_func', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('linear_act_func_out', [None, 'Tanh', 'Sigmoid']),        
    }
    ])

# set the optimize algorithm
optim_algo = tpe.suggest

# the the best options
def get_best():
    best = fmin(fn = o_func,
            space = li_space,
            algo = optim_algo,
            max_evals = 500)
    return best



def train_batch(model, loss_fn, optimizer, src, tgt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)

    optimizer.zero_grad()
    out = model(src)
    loss = loss_fn(out, tgt)
    #print(loss)

    loss.backward()
    optimizer.step()
    return True

def evaluate(model, src, tgt):
    arr1 = predict(model, src)
    arr2 = tgt.view(1, -1)
    #print(arr1)
    #print(arr2)
    arr = torch.cat((arr1, arr2), 0).numpy()
    corr = numpy.corrcoef(arr)[0][1]
    if math.isnan(corr):
        print("the corr is nan, which means the variance of the scores is 0")
        corr = 0
    print("the correlation coeffizient is : %f" %(corr))
    return corr

def predict(model, src):
    src = torch.autograd.Variable(src, requires_grad = False)
    out = model(src)
    return out.view(1, -1).data
