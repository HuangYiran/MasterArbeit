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
from LinearModel import BasicLinear, BasicLinear_dropout, BiLinear
from MaskedModel import MaskedModel1, MaskedModel2
from Params import Params
from hyperopt import fmin, tpe, hp
###########################################################################
# define the objective function, this function will be used in optimization
###########################################################################
def o_func(params):
    # set params
    opt = Params()
    opt.set_params(params)
    opt.show_params()

    # read data
    data = DataUtil(opt)
    data.normalize_minmax()

    # build model
    if opt.model == "BiLinear":
        model = BiLinear(dim2 = opt.dim2, act_func = opt.act_func, act_func_out = opt.act_func_out)
    elif opt.model == "BasicLinearDropout":
        model = BasicLinear_dropout(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "MaskedModel1":
        model = MaskedModel1(dim2 = opt.dim2, act_func = opt.act_func)
    elif opt.model == "MaskedModel2":
        model = MaskedModel2(dim2 = opt.dim2, act_func = opt.act_func)
    else:
        model = BasicLinear(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, mom = opt.momentum)
    print(model)

    # set optimizer and loss functioin, use if-else because their parameters are different.
    # lr = 0.002, betas = (0.9, 0.888), eps = 1e-08, weight_decay = 0
    if opt.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr, eps = opt.eps, weight_decay = opt.weight_decay)
    
    if opt.loss_fn == 'MSELoss':
        loss_fn = torch.nn.MSELoss()
    elif opt.loss_fn == 'L1Loss':
        loss_fn = torch.nn.L1Loss()

    # get the number of batch 
    nu_batch = data.get_nu_batch()
    print("number of batch is %d"%(nu_batch))

    # train
    for i in range(50 * nu_batch):
        src, tgt = data.get_batch_repeatly()
        if i % 10 == 0:
            print("evaluate %d" %(i/10))
            evaluate(model, src, tgt)
            evaluate_loss(model, loss_fn, src, tgt)
        else:
            train_batch(model, loss_fn, optimizer, src, tgt)
    
    out = 0.0
    for i in range(10):
        src, tgt = data.get_batch_repeatly()
        out = out + evaluate(model, src, tgt)
        
    return 1 - out/10.0

#######################################
# set the search space for linear Model 
#######################################
li_space = hp.choice('opt',[
    {
        'type': 'nonlinear',
        'model': '.BssicLinearModelDropout',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('basic_dp_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('basic_dp_optimizer', ['Adam']),
        'loss_fn': hp.choice('basic_dp_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('basic_batch_size', [50, 100, 200]),
        'lr': hp.uniform('basic_dp_lr', 0.001, 0.9),
        'dim2': hp.choice('basic_dp_dim2', [50, 100, 500]),
        'dim3': hp.choice('basic_dp_dim3', [None, 5, 10, 50]),
        'act_func': hp.choice('basic_dp_act_func', ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('basic_dp_act_func_out', ['Tanh', 'Sigmoid', None]),
        'drop_out_rate': hp.uniform('basic_dp_drop_out_rate', 0.2, 0.8),
    },
    {
        'type': 'nonlinear',
        'model': 'BssicLinearModel',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('basic_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('basic_optimizer', ['Adam']),
        'loss_fn': hp.choice('basic_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('basic_batch_size', [50, 100, 200]),
        'lr': hp.uniform('basic_lr', 0.001, 0.9),
        'dim2': hp.choice('basic_dim2', [50, 100, 500]),
        'dim3': hp.choice('basic_dim3', [None, 5, 10, 50]),
        'act_func': hp.choice('basic_act_func', ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('basic_act_func_out', ['Tanh', 'Sigmoid', None]),
#        'drop_out_rate': hp.uniform('basic_drop_out_rate', 0.2, 0.8),
        'momentum': hp.uniform('basic_bn_momentum', 0.1, 0.9),
    },
    {
        'type': 'nonlinear',
        'model': 'BiLinear',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('bi_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('bi_optimizer', ['Adam']),
        'loss_fn': hp.choice('bi_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('bi_batch_size', [50, 100, 200]),
        'lr': hp.uniform('bi_lr', 0.001, 0.9),
        'dim2': hp.choice('bi_dim2', [None, 10, 50]),
        'act_func': hp.choice('bl_act_func1', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('bl_act_func_out', [None, 'Tanh', 'Sigmoid']),        
    },
    {
        'type': 'masked',
        'model': 'MaskedModel1',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('bi_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('m1_optimizer', ['Adam']),
        'loss_fn': hp.choice('m1_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('m1_batch_size', [50, 100, 200]),
        'lr': hp.uniform('m1_lr', 0.001, 0.9),
        'dim2': hp.choice('m1_dim2', [10, 50]),
        'act_func': hp.choice('m1_act_func1', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
    },
    {
        'type': 'masked',
        'model': 'MaskedModel2',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('bi_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('m2_optimizer', ['Adam']),
        'loss_fn': hp.choice('m2_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('m2_batch_size', [50, 100, 200]),
        'lr': hp.uniform('m2_lr', 0.001, 0.9),
        'dim2': hp.choice('m2_dim2', [10, 50]),
        'act_func': hp.choice('m2_act_func1', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
    }
    ])

## separate
basic_linear_space = hp.choice('opt',[
    {
        'type': 'nonlinear',
        'model': './model/LinearModel',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('basic_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('basic_optimizer', ['Adam']),
        'loss_fn': hp.choice('basic_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('basic_batch_size', [50, 100, 200]),
        'lr': hp.uniform('basic_lr', 0.001, 0.9),
        'dim2': hp.choice('basic_dim2', [50, 100, 500]),
        'dim3': hp.choice('basic_dim3', [None, 5, 10, 50]),
        'act_func': hp.choice('basic_act_func', ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('basic_act_func_out', ['Tanh', 'Sigmoid', None]),
#        'drop_out_rate': hp.uniform('basic_drop_out_rate', 0.2, 0.8),
        'momentum': hp.uniform('basic_bn_momentum', 0.1, 0.9),
    }
])
basic_dp_linear_space =  hp.choice('opt',[
    {
        'type': 'nonlinear',
        'model': './model/LinearModel',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('basic_dp_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('basic_dp_optimizer', ['Adam']),
        'loss_fn': hp.choice('basic_dp_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('basic_batch_size', [50, 100, 200]),
        'lr': hp.uniform('basic_dp_lr', 0.001, 0.9),
        'dim2': hp.choice('basic_dp_dim2', [50, 100, 500]),
        'dim3': hp.choice('basic_dp_dim3', [None, 5, 10, 50]),
        'act_func': hp.choice('basic_dp_act_func', ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('basic_dp_act_func_out', ['Tanh', 'Sigmoid', None]),
        'drop_out_rate': hp.uniform('basic_dp_drop_out_rate', 0.2, 0.8),
    }
])
bi_linear_space = hp.choice('opt',[
    {
        'type': 'nonlinear',
        'model': 'BiLinear',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('bi_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('bi_optimizer', ['Adam']),
        'loss_fn': hp.choice('bi_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('bi_batch_size', [50, 100, 200]),
        'lr': hp.uniform('bi_lr', 0.001, 0.9),
        'dim2': hp.choice('bi_dim2', [None, 10, 50]),
        'act_func': hp.choice('bi_act_func1', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
        'act_func_out': hp.choice('bl_act_func_out', [None, 'Tanh', 'Sigmoid']),        
    }
])
m1_space = hp.choice('opt', [
    {
        'type': 'masked',
        'model': 'MaskedModel2',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('bi_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('m2_optimizer', ['Adam']),
        'loss_fn': hp.choice('m2_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('m2_batch_size', [50, 100, 200]),
        'lr': hp.uniform('m2_lr', 0.001, 0.9),
        'dim2': hp.choice('m2_dim2', [None, 10, 50]),
        'act_func': hp.choice('m2_act_func1', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
    }
])
m2_space = hp.choice('opt', [
    {
        'type': 'masked',
        'model': 'MaskedModel2',
        'src_sys': '../data/MasterArbeit/data/sys_hidden',
        'src_ref': '../data/MasterArbeit/data/ref_hidden',
        'tgt': hp.choice('bi_tgt', ['../data/MasterArbeit/data/data_scores', '../data/MasterArbeit/data/normalized_data_scores']),
        'optimizer': hp.choice('m1_optimizer', ['Adam']),
        'loss_fn': hp.choice('m1_loss_fn', ['MSELoss', 'L1Loss']),
        'batch_size': hp.choice('m2_batch_size', [50, 100, 200]),
        'lr': hp.uniform('m2_lr', 0.001, 0.9),
        'dim2': hp.choice('m2_dim2', [10, 50]),
        'act_func': hp.choice('m2_act_func1', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']),
    }
])
############################
# set the optimize algorithm
############################
optim_algo = tpe.suggest

######################
# get the best options
######################
def get_best():
    best = fmin(fn = o_func,
            space = li_space,
            algo = optim_algo,
            max_evals = 11)
    return best





###########
#training
###########

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

def evaluate_loss(model, loss_fn, src, tgt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)
    out = model(src)
    loss = loss_fn(out, tgt).data.numpy()
    mean = numpy.mean(loss)
    std = numpy.std(loss)
    print("the mean loss is %f, the std is %f" %(mean, std))
    return mean

def predict(model, src):
    src = torch.autograd.Variable(src, requires_grad = False)
    out = model(src)
    return out.view(1, -1).data
