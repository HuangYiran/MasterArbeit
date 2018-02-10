# -*- coding: UTF-8 -*-
import sys
sys.path.append("./utils/")
sys.path.append("./models/")
import torch
import argparse
import numpy
import random
import math
import os
import nnInit
import nnLoss

from data import DataUtil
from LinearModel import BasicLinear, BasicLinear_dropout, BiLinear
from MaskedModel import MaskedModel1, MaskedModel2
from FullHiddenModel import MultiHeadAttnMlpModel, MultiHeadAttnLSTMModel, MultiHeadAttnConvModel, MultiHeadAttnConvModel2, ScaledDotAttnConvModel
from RankModel import *
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
    
    # file to save the tmp result
    path_loss = opt.dir_mid_result + 'loss'
    path_corr = opt.dir_mid_result + 'corr'
    if os.path.exists(path_loss):
        os.remove(path_loss)
    if os.path.exists(path_corr):
        os.remove(path_corr)
    file_loss = open(path_loss, 'a')
    file_corr = open(path_corr, 'a')

    # read data and normalize if necessary
    data = DataUtil(opt)
    data.normalize_minmax()

    # build model, 
    model = build_model(opt)

    # if cuda is set use gpu to train the model
    if opt.cuda == "True": # don't know how to set boolean value in the hyperopt
        model.cuda()
        
    # set optimizer and loss functioin, use if-else because their parameters are different.
    # lr = 0.002, betas = (0.9, 0.888), eps = 1e-08, weight_decay = 0
    if opt.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr, eps = opt.eps, weight_decay = opt.weight_decay)   
    if opt.loss_fn == 'CorrLoss':
        loss_fn = nnLoss.CorrLoss()
    elif opt.loss_fn == 'L1Loss':
        loss_fn = torch.nn.L1Loss()
    elif opt.loss_fn == 'MSELoss':
        loss_fn = torch.nn.MSELoss()
    elif opt.loss_fn == 'MSECorrLoss':
        loss_fn = nnLoss.MSECorrLoss()
    elif opt.loss_fn == 'PReLULoss':
        loss_fn = nnLoss.PReLULoss(opt.PReLU_rate)
    elif opt.loss_fn == 'PReLUCorrLoss':
        loss_fn = nnLoss.PReLUCorrLoss()
    elif opt.loss_fn == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        print("++++++++ unknown Loss function, set loss to mse loss")    
    if opt.cuda == "True":
        loss_fn.cuda()

    # load checkpoint
    if opt.resume:
        if os.path.isfile(opt.checkpoint):
            print("loading checkpoint from '{}'".format(opt.checkpoint))
            # forget to check the model type
            checkpoint = torch.load(opt.checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        # if not resume from the checkpoint then initialize the parameters
        print 'initializing the model...'
        model.apply(nnInit.weight_init)
            
    # get the number of batch 
    nu_batch, nu_val_batch, nu_test_batch = data.get_nu_batch()
    print("number of batch is %d \nnumber of val batch %d \nnumber of test batch %d"%(nu_batch, nu_val_batch, nu_test_batch))

    # train
    for i in range(15 * nu_batch):
        if i % 10 == 0:
            # evaluate 
            src, tgt = data.get_val_batch(rep = True)
            if opt.cuda == "True":
                tgt = tgt.cuda()
            # src, tgt = data.get_batch(rep = True)
            print("evaluate %d" %(i/10))
            if opt.rank:
                corr = evaluate_corr_rank(model, src, tgt)
            else:
                corr = evaluate_corr(model, src, tgt)
            loss = evaluate_loss(model, loss_fn, src, tgt, opt)
            
            # write mid result
            tmp_loss = "%d,%f"%(int(i/10), loss)
            file_loss.write(tmp_loss)
            file_loss.write('\n')
            tmp_corr = "%d,%f"%(int(i/10), corr)
            file_corr.write(tmp_corr)
            file_corr.write('\n')
            
            # save checkpoint
            print "save checkpoint " + str(i/10)
            save_checkpoint({
                'model': opt.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, './checkpoints/cp'+str(i/10))
        else:
            # train
            if opt.isRandom:
                src, tgt = data.get_random_batch()
            else:
                src, tgt = data.get_batch(rep = True)
            if opt.cuda == "True":
                tgt = tgt.cuda
            train_batch(model, loss_fn, optimizer, src, tgt, opt)
    
    print "--test model"
    test = test_model(opt, model, data)
    print "--val model"
    val = val_model(opt, model, data)
    with open(opt.dir_mid_result + '_val', 'w') as fi:
        val = "%f"%(val)
        fi.write(val)
        fi.write('\n')
    file_corr.close()
    file_loss.close()
        
    return test, val


###############################################
# some function that extracted from the o_func
##############################################
def test_model(opt, model, data):
    out = 0.0
    path_test = opt.dir_mid_result + 'test'
    if os.path.exists(path_test):
        os.remove(path_test)
    file_test = open(path_test, 'a')
    num_train, num_val, num_test = data.get_nu_batch()
    for i in range(num_test):
        src, tgt = data.get_test_batch()
        if opt.cuda == "True":
            tgt = tgt.cuda()
        if opt.rank:
            corr = evaluate_corr_rank(model, src, tgt)
        else:
            corr = evaluate_corr(model, src, tgt)
        out = out + corr
        tmp_corr = "%d,%f"%(i, corr)
        file_test.write(tmp_corr)
        file_test.write('\n')
    file_test.close()
    return out/num_test

def val_model(opt, model, data):
    out = 0.0
    num_train, num_val, num_test = data.get_nu_batch()
    data.reset_cur_val_index()
    for i in range(num_val):
        src, tgt = data.get_val_batch()
        if opt.cuda == "True":
            tgt = tgt.cuda()
        if opt.rank:
            corr = evaluate_corr_rank(model, src, tgt)
        else:
            corr = evaluate_corr(model, src, tgt)
        out = out + corr
    return out/num_val

def build_model(opt):
    # build model, 
    if opt.model == "BiLinear":
        model = BiLinear(dim2 = opt.dim2, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "BasicLinearDropout":
        model = BasicLinear_dropout(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "MaskedModel1":
        model = MaskedModel1(dim2 = opt.dim2, act_func = opt.act_func)
    elif opt.model == "MaskedModel2":
        model = MaskedModel2(dim2 = opt.dim2, act_func = opt.act_func)
    elif opt.model == "ScaledDotAttnConvModel":
        model = ScaledDotAttnConvModel(d_rate_attn = opt.d_rate_attn, dim1 = opt.dim1, act_func1 = opt.act_func1, kernel_size1 = opt.kernel_size1, stride1 = opt.stride1, act_func2 = opt.act_func2, kernel_size2 = opt.kernel_size2, stride2 = opt.stride2)
    elif opt.model == "MultiHeadAttnMlpModel":
        model = MultiHeadAttnMlpModel(num_head = opt.num_head, num_dim_k = opt.num_dim_k, num_dim_v = opt.num_dim_v, d_rate_attn = opt.d_rate_attn, act_func1 = opt.act_func1, dim2 = opt.dim2, act_func2 = opt.act_func2)
    elif opt.model == "MultiHeadAttnLSTMModel":
        model = MultiHeadAttnLSTMModel(num_head = opt.num_head, num_dim_k = opt.num_dim_k, num_dim_v = opt.num_dim_v, d_rate_attn = opt.d_rate_attn, dim2 = opt.dim2, act_func2 = opt.act_func2)
    elif opt.model == "MultiHeadAttnConvModel":
        model = MultiHeadAttnConvModel(num_head = opt.num_head, num_dim_k = opt.num_dim_k, num_dim_v = opt.num_dim_v, d_rate_attn = opt.d_rate_attn, dim1 = opt.dim1, act_func1 = opt.act_func1, kernel_size1 = opt.kernel_size1, stride1 = opt.stride1, act_func2 = opt.act_func2, kernel_size2 = opt.kernel_size2, stride2 = opt.stride2)
    elif opt.model == "MultiHeadAttnConvModel2":
        model = MultiHeadAttnConvModel2(num_head = opt.num_head, num_dim_k = opt.num_dim_k, num_dim_v = opt.num_dim_v, d_rate_attn = opt.d_rate_attn, dim1 = opt.dim1, act_func1 = opt.act_func1, kernel_size1 = opt.kernel_size1, stride1 = opt.stride1, act_func2 = opt.act_func2, kernel_size2 = opt.kernel_size2, stride2 = opt.stride2)
    elif opt.model == "MLPRank":
        model = MLPRank(dim2 = opt.dim2, dim3 = opt.dim3, dim4 = opt.dim4, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "MLPSoftmaxRank":
        model = MLPSoftmaxRank(dim2 = opt.dim2, dim3 = opt.dim3, dim4 = opt.dim4, act_func = opt.act_func, d_rate = opt.drop_out_rate)
    elif opt.model == "TriLinearRank":
        model = TriLinearRank(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "TriLinearSoftmaxRank":
        model = TriLinearSoftmaxRank(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, d_rate = opt.drop_out_rate)
    elif opt.model == "MaskedModelRank1":
        model = MaskedModelRank1(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "MaskedModelRank2":
        model = MaskedModelRank2(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "MaskedModelRank3":
        model = MaskedModelRank3(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "BasicLinear":
        model = BasicLinear(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, mom = opt.momentum)
    print(model)
    return model




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
fullHiddenModel_space = hp.choice('opt', [
    {
        'tgt': '../data/MasterArbeit/data2/record_NIST_dev2015_clean', 
        'src_sys': '../data/MasterArbeit/data2/hidden_pred_preprodev2015',
        'src_ref': '../data/MasterArbeit/data2/hidden_ref_preprodev2015',
        'val_tgt': '../data/MasterArbeit/data2/record_NIST_tst2016_clean',
        'src_val_sys': '../data/MasterArbeit/data2/hidden_pred_preprotst2016',
        'src_val_ref': '../data/MasterArbeit/data2/hidden_ref_preprotst2016',
        'test_tgt': '../data/MasterArbeit/data2/record_NIST_tst2016_clean',
        'src_test_sys': '../data/MasterArbeit/data2/hidden_pred_preprotst2016',
        'src_test_ref': '../data/MasterArbeit/data2/hidden_ref_preprotst2016',
        'cuda': 'True',
        'model': hp.choice('model', ['MultiHeadAttnMlpModel', 'MultiHeadAttnLSTMModel', 'MultiHeadAttnConvModel']),
        'batch_size': hp.choice('batch_size', [20, 50, 100]),
        'num_head': hp.choice('num_head', [4, 8, 32, 64, 128]),
        'num_dim_k': hp.choice('num_dim_k', [16, 32, 64, 128, 512]),
        'num_dim_v': hp.choice('num_dim_v', [16, 32, 64, 128, 512]),
        'd_rate_attn': hp.uniform('d_rate_attn', 0.0, 0.9),
        #'act_func1': hp.choice('act_func1', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'])
        #'act_func2': hp.choice('act_func2', ['ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'])
        'act_func1': 'LeakyReLU',
        'act_func2': 'LeakyReLU',
        'dim1': hp.choice('dim1', [64, 128, 256]),
        'dim2': hp.choice('dim2', [16, 32, 64]),
        'kernel_size1': hp.choice('kernel_size1', [2, 4, 8, 32, 64]),
        'kernel_size2': hp.choice('kernel_size2', [2, 4, 8, 32]),
        'stride1': hp.choice('stride1', [1, 2, 4, 16]),
        'stride2': hp.choice('stride2', [1, 2, 4, 16])
    }])
############################
# set the optimize algorithm
############################
optim_algo = tpe.suggest

######################
# get the best options
######################
def get_best():
    best = fmin(fn = o_func,
            space = fullHiddenModel_space,
            algo = optim_algo,
            max_evals = 500)
    return best





###########
#training
###########

def train_batch(model, loss_fn, optimizer, src, tgt, opt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)
    if not opt.sf_output:
        tgt = tgt.unsqueeze(dim = 1)

    optimizer.zero_grad()
    model.train()
    out = model(src)
    loss = loss_fn(out, tgt)
    #print(loss)

    loss.backward()
    optimizer.step()
    return True

def evaluate_corr(model, src, tgt):
    arr1 = predict(model, src)
    arr1 = arr1.view(1, -1)
    arr2 = tgt.view(1, -1)
    #print(arr1)
    #print(arr2)
    arr = torch.cat((arr1, arr2), 0)
    arr = arr.cpu()
    arr = arr.numpy()
    corr = numpy.corrcoef(arr)[0][1]
    if math.isnan(corr):
        print("the corr is nan, which means the variance of the scores is 0")
        corr = 0
    print("the correlation coeffizient is : %f" %(corr))
    return corr

def evaluate_corr_rank(model, src, tgt, threshold = 0.5):
    arr1 = predict(model, src)
    arr1 = arr1.numpy()
    # arr2 = tgt.view(1, -1)
    arr2 = tgt.numpy()
    # precess the output of the model
    if arr1.shape[1] == 3:
        # softmax output
        arr1 = numpy.array(list(map(result_transform_sf_to_score, arr1)))
        arr2 = arr2 - 1
    else:
        # score output, didn't use the threshold either.
        arr1 = numpy.array(list(map(lambda x: round(x), arr1)))
    arr = numpy.vstack((arr1, arr2))
    corr = numpy.corrcoef(arr)[0][1]
    if math.isnan(corr):
        print("the corr is nan, which means the variance of the scores is 0")
        corr = 0
    print("the correlation coeffizient is : %f" %(corr))
    return corr

def result_transform_sf_to_score(x):
    a, b, c = x[0], x[1], x[2]
    if a > b and a > c:
        return -1
    elif b > a and b > c:
        return 0
    elif c > a and c > b:
        return 1
    else:
        # ???
        return 0
    
def get_result1(x):
    pass
     
        
def evaluate_loss(model, loss_fn, src, tgt , opt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)
    if not opt.sf_output:
        tgt = tgt.unsqueeze(dim = 1)
    model.eval()
    #print("src1",src[1][0])
    #print("src2",src[1][100]) 
    out = model(src)
    #print("shape out and tgt", out.data.shape, tgt.data.shape)
    loss = loss_fn(out, tgt).data
    loss = loss.cpu()
    loss = loss.numpy()
    mean = numpy.mean(loss)
#    std = numpy.std(loss)
    print("the mean loss is %f" %(mean))
    return mean

def predict(model, src):
    src = torch.autograd.Variable(src, requires_grad = False)
    model.eval()
    out = model(src)
    #return out.view(1, -1).data
    return out.data

def save_checkpoint(state, filename):
    #torch.save(state, filename)
    print "please remove the comment sign before saving a checkpoint"
