# -*- coding: UTF-8 -*-
import sys
sys.path.append('utils/')
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

class EinLayerRank(torch.nn.Module):
    """
    one layer linear model : 1500 -3
    """
    def __init__(self):
        super(EinLayerRank, self).__init__()
        #dim1 = 768
        dim1 = 1500
        self.li = torch.nn.Linear(dim1, 3)
        self.sf = torch.nn.LogSoftmax()
#        self.af = nnActi.get_acti('ReLU')

    def forward(self, input):
#        out = self.af(self.li(input))

        #inp = (input_s1-input_s2) * input_ref
        out = self.sf(self.li(input))
        return out

class ELMo(torch.nn.Module):
    """
    Elmo
    """
    def __init__(self, seq_len = 50, num_layers = 2, num_dim = 500):
        super(ELMo, self).__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_dim = num_dim
        #self.weight = torch.autograd.Variable(torch.FloatTensor(self.num_layers, self.seq_len), requires_grad = True)
        self.weight_layers = torch.autograd.Variable(torch.FloatTensor(self.num_layers), requires_grad = True)
        self.weight_seq = torch.autograd.Variable(torch.FloatTensor(self.seq_len), requires_grad = True)
        # build a mlp model
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1500, 500),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 3),
            torch.nn.LogSoftmax()
        )
        # softmax
        self.sf = torch.nn.Softmax()
     
    def forward(self, out_s1, out_s2, out_ref):
        # only use decoder hidden states
        #print('out_s1[1] %s \nout_s2[1] %s \nout_ref[1] %s'%(str(out_s1[1].shape), str(out_s2[1].shape), str(out_ref[1].shape)))
        input = [out_s1, out_s2, out_ref]
        input = torch.cat(input, 3) # (batch, seq_len, num_layer, num_dims): (bs, 50, 2, 1500)
        # print input.data.shape
        # expand weight so that it can do the bmm later, include self.weight_layers and self.weight_seq
        # get shape
        batch_size, seq_len, num_layers, num_dim = input.data.shape
        assert(seq_len == self.seq_len and num_layers == self.num_layers)
        exp_weight_layers =self.weight_layers.expand(batch_size, self.num_layers) # ==> (batch_size, num_layers)
        exp_weight_seq = self.weight_seq.expand(batch_size, self.seq_len) # ==> (batch_size, seq_len)
        # do the softmax for self.weight_layers, self.weight_seq
        exp_weight_layers = self.sf(exp_weight_layers).unsqueeze(1) # ==> (batch_size, 1, num_layers)
        exp_weight_seq = self.sf(exp_weight_seq).unsqueeze(1) # ==> (batch_size, 1, seq_len)
        # mul input
        # use inner state only
        data = input
        # weighted sum layer
        data = data.transpose(1,2).contiguous()
        data = data.view(batch_size, num_layers, -1) # ==> (batch_size, num_layers, seq_len * num_dim)
        ls_data = torch.bmm(exp_weight_layers, data).squeeze() # ==> (batch_size, seq_len * num_dim)
        ls_data = ls_data.view(batch_size, seq_len, -1) # ==> (batch_size, seq_len, num_dim)
        # weighted sum seq
        ss_data = torch.bmm(exp_weight_seq, ls_data).squeeze() # ==> (batch_size, num_dim)
        out = self.mlp(ss_data)
        return out

class TwoLayerRank(torch.nn.Module):
    """
    two layer linear model: 1500 - 64 - 3
    use softmax as loss function
    """
    def __init__(self, dim2 = 64, act_func = "ReLU", act_func_out = None, d_rate = 0.5, mom = 0.1):
        super(TwoLayerRank, self).__init__()
        dim1 = 1000
        #dim1 = 900
        #self.layers = torch.nn.Sequential()
        #self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        #self.layers.add_module("bn", torch.nn.BatchNorm1d(dim2, momentum = mom))
        #self.layers.add_module("act_func1", nnActi.get_acti(act_func))
        #self.layers.add_module("fc2", torch.nn.Linear(dim2, 3))
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.bn = torch.nn.BatchNorm1d(dim2, momentum = mom)
        self.af = nnActi.get_acti(act_func)
        self.fc2 = torch.nn.Linear(dim2, 3)

    def forward(self, input):
        """
        input: (batch_size, 1500): (s1, s2, ref)
        output (batch_size, 3) s1 win, s2 win, equal
        """
        #out = self.layers(input)
        input_s1 = input[:, :500]
        input_s2 = input[:, 500: 1000]
        input_ref = input[:, 1000:]

        inp = input[:,:1000]
        #inp = input (0.06, 0.404, 0.397)
        #inp = (input_s1-input_s2) * input_ref #(0.123, 0.426, 0.442)
        #inp = input_ref*2 - input_s1 - input_s2 (0.004, 0.137, 0.139)
        #inp = (input_s1 * input_s2) - input_ref (0.002, 0.190, 0.187)
        #inp = (input_s1 - input_ref) * input_s2 (0.079, 0.366, 0.365)
        #inp = input_s1 - input_s2 (0.08, 0.417, 0.424)
        #inp = input_ref - input_s2  (0.038, 0.354, 0.354)
        #inp = input_ref * input_s2 (0.002, 0.260, 0.257)
        #inp = input_s1 * input_s2 * input_ref (-0.013, 0.200, 0.196)
        #inp = input_s1 * input_s2 (-0.22, 0.173, 0.172)
        #inp = (input_s1 - input_s2)* (input_s1 -input_s2) * input_ref (-0.010, 0.136, 0.144)
        #inp = input_s1 + input_s2 + input_refa (-0.005, 0.166, 0.165)
        #inp = input_s1 * input_s2 * input_ref (-0.018, 0.201, 0.197)
        #inp = input_s1*input_ref + input_s2*input_ref (0.002, 0.222, 0.215)
        #t = torch.stack([input_s1, input_s2, input_ref], dim = 1)
        #t = t.unsqueeze(dim = 1)
        #inp = self.mp(t).squeeze()
        out = self.fc2(self.af(self.bn(self.fc1(inp))))
        return out

class TwoLayerRank2(torch.nn.Module):
    """
    two layer linear model: 1500 - 64 - 1
    use mse as loss function
    """
    def __init__(self, dim2 = 64, act_func = "ReLU", act_func_out = None, d_rate = 0.5, mom = 0.1):
        super(TwoLayerRank2, self).__init__()
        #dim1 = 1500
        dim1 = 192 
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

class TwoLayerRank3(torch.nn.Module):
    """
    The model from the tutor
    two layer linear model: 1500-500-3 with as active function
    """
    def __init__(self, dim2 = 500):
        super(TwoLayerRank3, self).__init__()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(1500, dim2),
                torch.nn.Dropout(0.5),
                torch.nn.BatchNorm1d(dim2),
#                torch.nn.Tanh(),
                torch.nn.ReLU(),
                torch.nn.Linear(dim2, 3),
                torch.nn.LogSoftmax()
                )
    
    def forward(self, input):
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
        dim1 = 1500
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

        self.layers.add_module("softmax", torch.nn.LogSoftmax())


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

