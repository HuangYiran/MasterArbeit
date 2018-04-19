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

from scipy import stats
from data import DataUtil
from LinearModel import BasicLinear, BasicLinear_dropout, BiLinear, TwoLayerLinear
from MaskedModel import MaskedModel1, MaskedModel2, MaskedModel3
from FullHiddenModel import *
from RankModel import *
from Params import Params
from hyperopt import fmin, tpe, hp
from valuation import valTauLike
from transform import daToRr
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
    elif opt.optim == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters())
    elif opt.optim == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters())
    elif opt.optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters())
    elif opt.optim == 'Rprop':
        optimizer = torch.optim.Rprop(model.parameters())
    elif opt.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)
    else:
        print 'unrecognized optim function, use Adam instead'
        optimizer = torch.optim.Adam(model.parameters())
 
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
    for i in range(10 * nu_batch):
        if i%nu_batch == 0:
            # write mid result
            corr, taul = val_model(opt, model, data)
            test = test_model(opt, model, data)
            tmp_loss = "%d,%f"%(int(i/nu_batch), test)
            file_loss.write(tmp_loss)
            file_loss.write('\n')
            tmp_corr = "%d,%f"%(int(i/nu_batch), taul)
            file_corr.write(tmp_corr)
            file_corr.write('\n')
        if i % 10 == 0:
            # evaluate 
            src, tgt = data.get_val_batch(rep = True)
            if opt.cuda == "True":
                tgt = tgt.cuda()
            # src, tgt = data.get_batch(rep = True)
            print("evaluate %d" %(i/10))
            if opt.rank:
                corr = evaluate_tau_like(model, src, tgt)
#                corr = evaluate_corr(model, src, tgt)
            else:
                corr = evaluate_corr(model, src, tgt)
            loss = evaluate_loss(model, loss_fn, src, tgt, opt)
            
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
    
    # close the opened file
    file_corr.close()
    file_loss.close()

    # test and valuate the model 
    print "--test model"
    test = test_model(opt, model, data)
    print "--val model"
    if opt.rank:
        corr, taul = val_model(opt, model, data)
        return test, taul, corr
    else:
        val = val_model(opt, model, data)
        print '--test with rr data'
        test_rr = 0
        test_rr = test_da_model_with_rr_data(opt, model, data)
        return test, test_rr, val


###############################################
# some function that extracted from the o_func
##############################################
def test_model(opt, model, data):
    out_corr = 0.0
    out_mae = 0.0
    out_rmse = 0.0
    out_spearmanr = 0.0
    path_test = opt.dir_mid_result + 'test'
    if os.path.exists(path_test):
        os.remove(path_test)
    file_test = open(path_test, 'a')
    num_train, num_val, num_test = data.get_nu_batch()
    if not opt.rank:
        src = []
        tgt = []
#    if True:
        for i in range(num_test):
            tmp_src, tmp_tgt = data.get_test_batch(rep = True)
            src.append(tmp_src)
            tgt.append(tmp_tgt)
            if opt.cuda == "True":
                tmp_tgt = tmp_tgt.cuda()
            corr = evaluate_corr(model, tmp_src, tmp_tgt)
            out_corr = out_corr + corr
            tmp_corr = "%d,%f"%(i, corr)
            file_test.write(tmp_corr)
            file_test.write('\n')
        src = torch.cat(src)
        tgt = troch.cat(tgt)
        out_corr = evaluate_corr(model, src, tgt)
    else:
        src = []
        tgt = []
        for i in range(num_test):
            tmp_src, tmp_tgt = data.get_test_batch(rep = True)
            src.append(tmp_src)
            tgt.append(tmp_tgt)
        src = torch.cat(src)
        tgt = torch.cat(tgt)
        print(src.shape, tgt.shape)
        taul = evaluate_tau_like(model, src, tgt)
        return taul
    file_test.close()
#    return out_corr/num_test, out_mae/num_test, out_rmse/num_test, out_spearmanr/num_test
    return out_corr

def test_da_model_with_rr_data(opt, model, data):
    """
    train a da model and test this model with rr data,
        use the s1+ref and s2+ref to get the scores seperantly and compare the score the get the darr result
    input:
        model: a da model
        src: da data 
        tgt: rr data
    """
    # reload the data to do the test
    tgt ="../data/MasterArbeit/plan_c_rank_de.en/train_result"
    src_sys ="../data/MasterArbeit/plan_c_rank_de.en/train_s1_hidden"
    src_sys2 ="../data/MasterArbeit/plan_c_rank_de.en/train_s2_hidden"
    src_ref ="../data/MasterArbeit/plan_c_rank_de.en/train_ref_hidden"
    tgt_val ="../data/MasterArbeit/plan_c_rank_de.en/train_result"
    src_val_sys ="../data/MasterArbeit/plan_c_rank_de.en/train_s1_hidden"
    src_val_sys2 ="../data/MasterArbeit/plan_c_rank_de.en/train_s2_hidden"
    src_val_ref ="../data/MasterArbeit/plan_c_rank_de.en/train_ref_hidden"
    tgt_test ="../data/MasterArbeit/plan_c_rank_de.en/test_result"
    src_test_sys ="../data/MasterArbeit/plan_c_rank_de.en/test_s1_hidden"
    src_test_sys2 ="../data/MasterArbeit/plan_c_rank_de.en/test_s2_hidden"
    src_test_ref ="../data/MasterArbeit/plan_c_rank_de.en/test_ref_hidden"
    rank = True
    sf_output = False
    data.reload_data(src_sys, src_sys2, src_ref, tgt, src_val_sys, src_val_sys2, src_val_ref, tgt_val, src_test_sys, src_test_sys2, src_test_ref, tgt_test, rank, sf_output)
    # compute the scores 
    num_train, num_val, num_test = data.get_nu_batch()
    if not rank:
#    if True:
        for i in range(num_test):
            src, tgt = data.get_test_batch()
            if opt.cuda == "True":
                tgt = tgt.cuda()
            corr = evaluate_corr(model, src, tgt)
            out_corr = out_corr + corr
            tmp_corr = "%d,%f"%(i, corr)
            file_test.write(tmp_corr)
            file_test.write('\n')
    else:
        src = []
        tgt = []
        for i in range(num_test):
            tmp_src, tmp_tgt = data.get_test_batch()
            src.append(tmp_src)
            tgt.append(tmp_tgt)
        src = torch.cat(src)
        tgt = torch.cat(tgt)
        print(src.shape, tgt.shape)
        ref, s1, s2 = src.split(500, 1)
        src1 = torch.cat((s1, ref), 1)
        src2 = torch.cat((s2, ref), 1)
        o1 = predict(model, src1) 
        o2 = predict(model, src2)
        print(o1.shape, o2.shape)
        o1 = o1.squeeze().numpy().tolist()
        o2 = o2.squeeze().numpy().tolist()
        rr = daToRr(o1, o2, 0)
        taul = valTauLike(tgt, rr)
    return taul




def val_model(opt, model, data):
    out_corr = 0.0
    out_mae = 0.0
    out_rmse = 0.0    
    out_spearmanr = 0.0
    num_train, num_val, num_test = data.get_nu_batch()
    data.reset_cur_val_index()
    src_tau = []
    tgt_tau = []
    for i in range(num_val):
        src, tgt = data.get_val_batch()
        src_tau.append(src)
        tgt_tau.append(tgt)
        if opt.cuda == "True":
            tgt = tgt.cuda()
        if opt.rank:
            corr = evaluate_corr_rank(model, src, tgt)
        else:
            corr = evaluate_corr(model, src, tgt)
        out_corr = out_corr + corr
    src_tau = torch.cat(src_tau)
    tgt_tau = torch.cat(tgt_tau)
    if opt.rank:
        taul = evaluate_tau_like(model, src_tau, tgt_tau)
        return out_corr/num_val, taul
    else:
        return out_corr/num_val

def build_model(opt):
    # build model, 
    if opt.model == "TwoLayerLinear":
        model = TwoLayerLinear(act_func = opt.act_func, mom = opt.momentum)
    elif opt.model == "BiLinear":
        model = BiLinear(dim2 = opt.dim2, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate = opt.drop_out_rate)
    elif opt.model == "BasicLinearDropout":
        model = BasicLinear_dropout(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, act_func_out = opt.act_func_out, d_rate1 = opt.drop_out_rate, d_rate2 = opt.drop_out_rate2)
    elif opt.model == "Conv2dMlpModel_rank":
        model = Conv2dMlpModel_rank()
    elif opt.model == "Conv3dMlpModel_rank":
        model = Conv3dMlpModel_rank()
    elif opt.model == "LSTMMlpModel_rank":
        model = LSTMMlpModel_rank(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, softmax = opt.sf_output)
    elif opt.model == "MaskedModel1":
        model = MaskedModel1(dim2 = opt.dim2, act_func = opt.act_func)
    elif opt.model == "MaskedModel2":
        model = MaskedModel2(dim2 = opt.dim2, act_func = opt.act_func)
    elif opt.model == "MaskedModel3":
        model = MaskedModel3(dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, d_rate = opt.drop_out_rate, mom = opt.momentum)
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
    elif opt.model == "TwoLayerRank":
        model = TwoLayerRank(dim2 = opt.dim2, act_func = opt.act_func, mom = opt.momentum)
    elif opt.model == "TwoLayerRank2":
        model = TwoLayerRank2(dim2 = opt.dim2, act_func = opt.act_func, mom = opt.momentum)
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
    elif opt.model == "MLPSoftmaxDropoutRank":
        model = MLPSoftmaxDropoutRank(dim2 = opt.dim2, dim3 = opt.dim3, dim4 = opt.dim4, act_func = opt.act_func, d_rate1 = opt.drop_out_rate, d_rate2 = opt.drop_out_rate2, d_rate3 = opt.drop_out_rate3)
    elif opt.model == "ScaledDotAttnMlpModel_rank":
        model = ScaledDotAttnMlpModel_rank(d_rate_attn = opt.d_rate_attn, dim2 = opt.dim2, dim3 = opt.dim3, act_func = opt.act_func, softmax = opt.sf_output)
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
            space = m2_space,
            algo = optim_algo,
            max_evals = 5)
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

##############
#val and test
##############
def evaluate_corr(model, src, tgt):
    arr1 = predict(model, src)
    arr1 = arr1.view(1, -1)
    arr2 = tgt.view(1, -1)
    arr2 = arr2.float()
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

def evaluate_tau_like(model, src, tgt):
    """
    arr1 is the output of the model, and arr2 is the tgt
    so we should put the arr1 in the second parameter seat.
    """
    arr1 = predict(model, src)
    arr1 = arr1.numpy()
    arr2 = tgt.numpy()
    if arr1.shape[1] == 3:
        arr1 = list(map(result_transform_sf_to_score, arr1))
        arr2 = arr2 - 1
    else:
        arr1 = list(map(lambda x: round(x), arr1))
    taul = valTauLike(arr2, arr1)
    return taul

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

def evaluate_MAE(model, src, tgt, opt):
    loss_fn = torch.nn.L1Loss()
    mae = evaluate_loss(model, loss_fn, src, tgt, opt)
    return mae

def evaluate_RMSE(model, src, tgt, opt):
    loss_fn = torch.nn.MSELoss()
    mse = evaluate_loss(model, loss_fn, src, tgt, opt)
    rmse = math.sqrt(mse)
    return rmse

def evaluate_loss(model, loss_fn, src, tgt , opt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)
    if not opt.sf_output:
        tgt = tgt.unsqueeze(dim = 1)
    model.eval()
    # print("src1",src[1][0])
    # print("src2",src[1][100]) 
    out = model(src)
    #print("shape out and tgt", out.data.shape, tgt.data.shape)
    loss = loss_fn(out, tgt).data
    loss = loss.cpu()
    loss = loss.numpy()
    mean = numpy.mean(loss)
#   std = numpy.std(loss)
    print("the mean loss is %f" %(mean))
    return mean

def predict(model, src):
    """
    return type of Tensor
    """
    src = torch.autograd.Variable(src, requires_grad = False)
    model.eval()
    out = model(src)
    #return out.view(1, -1).data
    return out.data

def save_checkpoint(state, filename):
    #torch.save(state, filename)
    print "please remove the comment sign before saving a checkpoint"
