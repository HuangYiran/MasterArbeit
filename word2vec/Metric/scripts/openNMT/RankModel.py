# -*- coding: UTF-8 -*-
import sys
sys.path.append('../utils/')
import torch
import numpy
import nnActi
from MaskedModel import *
from LinearModel import *

class DaMlpRank(torch.nn.Module):
    """
    the model is consist of two da model 
    """
    def __init__(self, dim2 = 64, act_func = 'SeLU', softmax = False):
        super(DaMlpRank, self).__init__()
        self.num_dim = 300
        self.m1 = BasicLinear(dim2 = dim2, act_func = act_func, num_dim = self.num_dim)
        self.m2 = BasicLinear(dim2 = dim2, act_func = act_func, num_dim = self.num_dim)
        if softmax:
            self.li = torch.nn.Linear(2,3)
        else:
            self.li = torch.nn.Linear(2,1)

    def forward(self, input):
        input_s1 = input[:, :self.num_dim]
        input_s2 = input[:, self.num_dim: 2*self.num_dim]
        input_ref = input[:, 2*self.num_dim:]
        input_rs1 = torch.cat((input_s1, input_ref), 1)
        input_rs2 = torch.cat((input_s2, input_ref), 1)
        m1_out = self.m1(input_rs1)
        m2_out = self.m2(input_rs2)
        m_out = torch.cat((m1_out, m2_out), 1)
        out = self.li(m_out)
        return out

class TwoLayerRank(torch.nn.Module):
    """
    two layer linear model: 1500 - 64 - 3
    use softmax as loss function
    """
    def __init__(self, dim2 = 64, act_func = "ReLU", act_func_out = None, d_rate = 0.5, mom = 0.1):
        super(TwoLayerRank, self).__init__()
        #dim1 = 1500
        dim1 = 900
        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        self.layers.add_module("bn", torch.nn.BatchNorm1d(dim2, momentum = mom))
        self.layers.add_module("act_func1", nnActi.get_acti(act_func))
        self.layers.add_module("fc2", torch.nn.Linear(dim2, 3))

    def forward(self, input):
        """
        input: (batch_size, 1500): (s1, s2, ref)
        output (batch_size, 3) s1 win, s2 win, equal
        """
        out = self.layers(input)
        return out

class TwoLayerRank2(torch.nn.Module):
    """
    two layer linear model: 1500 - 64 - 1
    use mse as loss function
    """
    def __init__(self, dim2 = 64, act_func = "ReLU", act_func_out = None, d_rate = 0.5, mom = 0.1):
        super(TwoLayerRank2, self).__init__()
        #dim1 = 1500
        dim1 = 900
        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        self.layers.add_module("bn", torch.nn.BatchNorm1d(dim2, momentum = mom))
        self.layers.add_module("act_func2", nnActi.get_acti(act_func))
        self.layers.add_module("fc2", torch.nn.Linear(dim2, 1))
    
    def forward(self, input):
        """
        input: (batch_size, 1500): (s1, s2, ref)
        output: (batch_size) score
        """
        out = self.layers(input)
        return out

class MLPRank(torch.nn.Module):
    """
    multi-layer linear model: 1500 - dim2 -dim3 - 1
    use mse or corr loss
    """
    def __init__(self, dim2 = 500, dim3 = 64, dim4 = None,  act_func = "ReLU", act_func_out = None, d_rate = 0.5, mom = 0.1):
        super(MLPRank, self).__init__()
        #dim1 = 1500
        dim1 = 900
        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2, bias = False))
        self.layers.add_module("bn", torch.nn.BatchNorm1d(dim2))
        self.layers.add_module(act_func + "1", nnActi.get_acti(act_func))

        self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3, bias = False))
        self.layers.add_module("bn2", torch.nn.BatchNorm1d(dim3, momentum = mom))
        self.layers.add_module(act_func + "2", nnActi.get_acti(act_func))

        if dim4:
            self.layers.add_module("fc3", torch.nn.Linear(dim3, dim4, bias = False))
            self.layers.add_module("bn3", torch.nn.BatchNorm1d(dim4))
            self.layers.add_module(act_func + "3", nnActi.get_acti(act_func))
            self.layers.add_module("fc4", torch.nn.Linear(dim4, 1))
        else:
            self.layers.add_module("fc3", torch.nn.Linear(dim3, 1))

        if act_func_out:
            self.layers.add_module("bn_out", torch.nn.BatchNorm1d(1))
            self.layers.add_module("act_func_out", nnActi.get_acti(act_func_out))

    def forward(self, input):
        """
        input: (batch_size, 1500): (s1, s2, ref)
        output (1)
        """
        out = self.layers(input)
        return out

class MLPSoftmaxRank(torch.nn.Module):
    """
    multi-layer linear model: 1500 - dim2 - dim3 - 3
    use softmax to choose the result
    """
    def __init__(self, dim2 = 500, dim3 = 64, dim4 = None,  act_func = "ReLU", d_rate = 0.5, mom = 0.1):
        super(MLPSoftmaxRank, self).__init__()
        dim1 = 900
        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2, bias = False))
        self.layers.add_module("bn", torch.nn.BatchNorm1d(dim2))
        self.layers.add_module(act_func + "1", nnActi.get_acti(act_func))

        self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3, bias = False))
        self.layers.add_module("bn2", torch.nn.BatchNorm1d(dim3, momentum = mom))
        self.layers.add_module(act_func + "2", nnActi.get_acti(act_func))

        if dim4:
            self.layers.add_module("fc3", torch.nn.Linear(dim3, dim4, bias = False))
            self.layers.add_module("bn3", torch.nn.BatchNorm1d(dim4))
            self.layers.add_module(act_func + "3", nnActi.get_acti(act_func))
            self.layers.add_module("fc4", torch.nn.Linear(dim4, 3))
        else:
            self.layers.add_module("fc3", torch.nn.Linear(dim3, 3))

        self.layers.add_module("softmax", torch.nn.Softmax())


    def forward(self, input):
        """
        input: (batch_size, 1500): (s1, s2, ref)
        output (batch_size, 3) s1 win, s2 win, equal
        """
        out = self.layers(input)
        return out

class MLPSoftmaxDropoutRank(torch.nn.Module):
    """
    try to improve the generality
    """
    def __init__(self, dim2 = 500, dim3 = 64, dim4 = None, act_func = "ReLU", d_rate1 = 0.6, d_rate2 = 0.4, d_rate3 = 0.2, mom = 0.1):
        super(MLPSoftmaxDropoutRank, self).__init__()
        dim1 = 900
        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2, bias = False))
        #self.layers.add_module("dp1", torch.nn.Dropout(d_rate1))
        self.layers.add_module("bn1", torch.nn.BatchNorm1d(dim2, momentum = mom))
        self.layers.add_module(act_func + "1", nnActi.get_acti(act_func))

        self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3, bias = False))
        #self.layers.add_module("dp2", torch.nn.Dropout(d_rate2))
        self.layers.add_module("bn2", torch.nn.BatchNorm1d(dim3, momentum = mom))
        self.layers.add_module(act_func + "2", nnActi.get_acti(act_func))

        if dim4:
            self.layers.add_module("fc3", torch.nn.Linear(dim3, dim4, bias = False))
            self.layers.add_module("dp3", torch.nn.Dropout(d_rate3))
            self.layers.add_module(act_func + "3", nnActi.get_acti(act_func))
            self.layers.add_module("fc4", torch.nn.Linear(dim4, 3))
        else:
            self.layers.add_module("fc3", torch.nn.Linear(dim3, 3))

        # self.layers.add_module("softmax", torch.nn.Softmax())
    
    def forward(self, input):
        out = self.layers(input)
        return out

class TriLinearRank(torch.nn.Module):
    def __init__(self, dim2 = 64, dim3 = None, act_func = 'Tanh', d_rate = 0.5, act_func_out = None):
        super(TriLinearRank, self).__init__()
        dim1 = 300
        self.li_s1 = torch.nn.Linear(dim1, dim1, bias = False)
        self.li_s2 = torch.nn.Linear(dim1, dim1, bias = False)
        self.li_ref = torch.nn.Linear(dim1, dim1, bias = False)
        self.act_func = nnActi.get_acti(act_func)

        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        self.laeyrs.add_module("act_func1", nnActi.get_acti(act_func))
        self.layers.add_module("dp1", torch.nn.Dropout(d_rate))
        if dim3:
            self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3))
            self.layers.add_module("act_func2", nnActi.get_acti(act_func))
            self.layers.add_module("dp2", torch.nn.Dropout(d_rate))
            self.layers.add_module("fc_out", torch.nn.Linear(dim3, 1))
        else:
            self.layers.add_module("fc_out", torch.nn.Linear(dim2, 1))

        if act_func_out:
            self.layers.add_module("act_out", nnActi.get_acti(act_func_out))

    def forward(self, input):
        """
        input: (batch_size, 1500)
        output: score (1)
        """
        input_s1 = input[:, :500]
        input_s2 = input[:, 500: 1000]
        input_ref = input[:, 1000:]
        proj_s1 = self.li_s1(input_s1)
        proj_s2 = self.li_s2(input_s2)
        proj_ref = self.li_ref(input_ref)
        sum_in = proj_s1 + proj_s2 + proj_ref
        acted_sum_in = self.act_func(sum_in)
        out = self.layers(acted_sum_in)
        return out

class TriLinearSoftmaxRank(torch.nn.Module):
    def __init__(self, dim2 = 64, dim3 = None, act_func = 'Tanh', d_rate = 0.5):
        super(TriLinearSoftmaxRank, self).__init__()
        dim1 = 500
        self.li_s1 = torch.nn.Linear(dim1, dim1, bias = False)
        self.li_s2 = torch.nn.Linear(dim1, dim1, bias = False)
        self.li_ref = torch.nn.Linear(dim1, dim1, bias = False)
        self.act_func = nnActi.get_acti(act_func)

        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        self.layers.add_module("act_func1", nnActi.get_acti(act_func))
        self.layers.add_module("dp1", torch.nn.Dropout(d_rate))
        if dim3:
            self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3))
            self.layers.add_module("act_func2", nnActi.get_acti(act_func))
            self.layers.add_module("dp2", torch.nn.Dropout(d_rate))
            self.layers.add_module("fc_out", torch.nn.Linear(dim3, 3))
        else:
            self.layers.add_module("fc_out", torch.nn.Linear(dim2, 3))
        self.layers.add_module("softmax", torch.nn.Softmax())


    def forward(self, input):
        """
        input: (batch_size, 1500)
        output: (batch_size, 3)
        """
        input_s1 = input[:, :500]
        input_s2 = input[:, 500: 1000]
        input_ref = input[:, 1000:]
        proj_s1 = self.li_s1(input_s1)
        proj_s2 = self.li_s2(input_s2)
        proj_ref = self.li_ref(input_ref)
        sum_in = proj_s1 + proj_s2 + proj_ref
        acted_sum_in = self.act_func(sum_in)
        out = self.layers(acted_sum_in)
        return out

class MaskedModelRank1(torch.nn.Module):
    """
    not all the dimension in the hidden vector are important, so we create a mask in order to deside, which dimension is more important, which is less
    in this model, we only use the ref value to get the mask
    """
    def __init__(self, dim2  = 10, dim3 = None, act_func = 'Tanh', d_rate = 0.5, act_func_out = None):
        super(MaskedModelRank1, self).__init__()
        self.li_mask = torch.nn.Linear(500, 500)
        self.sf = torch.nn.Softmax()
        self.layers = torch.nn.Sequential()
        self.layers.add_module('fc1', torch.nn.Linear(1000, dim2))
        self.layers.add_module('act_fun1', nnActi.get_acti(act_func))
        self.layers.add_module('dp1', torch.nn.Dropout(d_rate))
        if dim3:
            self.layers.add_module('fc2', torch.nn.Linear(dim2, dim3))
            self.layers.add_module('act_func2', nnActi.get_acti(act_func))
            self.layers.add_module('dp2', torch.nn.Dropout(d_rate))
            self.layers.add_module('fc3', torch.nn.Linear(dim3, 1))
        else:
            self.layers.add_module('fc2', torch.nn.Linear(dim2, 1))
        if act_func_out:
            self.layers.add_module("act_out", nnActi.get_acti(act_func_out))

    def forward(self, input):
        """
        input: (batch_size, 1500)
        output: score(1)
        """
        input_s1 = input[:, :500]
        input_s2 = input[:, 500: 1000]
        input_ref = input[:, 1000:]
        proj_ref = self.li_mask(input_ref)
        mask = self.sf(proj_ref)
        masked_s1 = mask * input_s1
        masked_s2 = mask * input_s2
        masked_s = torch.cat((masked_s1, masked_s2), 1)
        out = self.layers(masked_s)

class MaskedModelRank2(torch.nn.Module):
    """
    not all the dimension in the hidden vector are important, so we create a mask in order to deside, which dimension is more important, which is less
    in this model, we only use the ref, s1, s2 value to get the mask
    """
    def __init__(self, dim2  = 10, dim3 = None, act_func = 'Tanh', d_rate = 0.5, act_func_out = None):
        super(MaskedModelRank2, self).__init__()
        self.li_mask = torch.nn.Linear(1500, 500)
        self.sf = torch.nn.Softmax()
        self.layers = torch.nn.Sequential()
        self.layers.add_module('fc1', torch.nn.Linear(1000, dim2))
        self.layers.add_module('act_fun1', nnActi.get_acti(act_func))
        self.layers.add_module('dp1', torch.nn.Dropout(d_rate))
        if dim3:
            self.layers.add_module('fc2', torch.nn.Linear(dim2, dim3))
            self.layers.add_module('act_func2', nnActi.get_acti(act_func))
            self.layers.add_module('dp2', torch.nn.Dropout(d_rate))
            self.layers.add_module('fc3', torch.nn.Linear(dim3, 1))
        else:
            self.layers.add_module('fc2', torch.nn.Linear(dim2, 1))
        if act_func_out:
            self.layers.add_module("act_out", nnActi.get_acti(act_func_out))

    def forward(self, input):
        """
        input: (batch_size, 1500)
        output: score(1)
        """
        input_s1 = input[:, :500]
        input_s2 = input[:, 500: 1000]
        input_ref = input[:, 1000:]
        proj_ref = self.li_mask(input)
        mask = self.sf(proj_ref)
        masked_s1 = mask * input_s1
        masked_s2 = mask * input_s2
        masked_s = torch.cat((masked_s1, masked_s2), 1)
        out = self.layers(masked_s)

class MaskedModelRank3(torch.nn.Module):
    """
    not all the dimension in the hidden vector are important, so we create a mask in order to deside, which dimension is more important, which is less
    in this model we use to masks getting from (ref, s1) and (ref, s2)
    """
    def __init__(self, dim2  = 10, dim3 = None, act_func = 'Tanh', d_rate = 0.5, act_func_out = None):
        super(MaskedModelRank3, self).__init__()
        self.li_mask1 = torch.nn.Linear(1000, 500)
        self.li_mask2 = torch.nn.Linear(1000, 500)
        self.sf1 = torch.nn.Softmax()
        self.sf2 = torch.nn.Softmax()
        self.layers = torch.nn.Sequential()
        self.layers.add_module('fc1', torch.nn.Linear(1000, dim2))
        self.layers.add_module('act_fun1', nnActi.get_acti(act_func))
        #self.layers.add_module('dp1', torch.nn.Dropout(d_rate))
        if dim3:
            self.layers.add_module('fc2', torch.nn.Linear(dim2, dim3))
            self.layers.add_module('act_func2', nnActi.get_acti(act_func))
            #self.layers.add_module('dp2', torch.nn.Dropout(d_rate))
            self.layers.add_module('fc3', torch.nn.Linear(dim3, 3))
        else:
            self.layers.add_module('fc2', torch.nn.Linear(dim2, 3))
        if act_func_out:
            self.layers.add_module("act_out", nnActi.get_acti(act_func_out))

    def forward(self, input):
        """
        input: (batch_size, 1500)
        output: score(1)
        """
        input_s1 = input[:, :500]
        input_s2 = input[:, 500: 1000]
        input_ref = input[:, 1000:]
        input_rs1 = torch.cat((input_s1, input_ref), 1)
        input_rs2 = torch.cat((input_s2, input_ref), 1)
        #print input_rs1
        proj_ref1 = self.li_mask1(input_rs1)
        proj_ref2 = self.li_mask2(input_rs2)
        #print input_ref1
        mask1 = self.sf1(proj_ref1)
        mask2 = self.sf2(proj_ref2)
        #print mask
        masked_s1 = mask1 * input_s1
        masked_s2 = mask2 * input_s2
        masked_s = torch.cat((masked_s1, masked_s2), 1)
        out = self.layers(masked_s)
        #print out.data.shape
        return out

