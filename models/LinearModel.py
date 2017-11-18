# -*- coding: UTF-8 -*-
import torch
import numpy

class BasicLinear(torch.nn.Module):
    """
    multi-layer linear model: 1000 - 500 - 1 
    """
    def __init__(self, dim2 = 500, dim3 = None, act_func = "ReLU", act_func_out = None, d_rate = 0.5, mom = 0.1):
        super(BasicLinear, self).__init__()
        dim1 = 1000
        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        self.layers.add_module(act_func + "1", getattr(torch.nn, act_func)())
        self.layers.add_module("bn", torch.nn.BatchNorm1d(dim2))
 
        if dim3:
            self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3))
            self.layers.add_module(act_func + "2", getattr(torch.nn, act_func)())
            self.layers.add_module("bn2", torch.nn.BatchNorm1d(dim3, momentum = mom))
#            self.layers.add_module("drop_out", torch.nn.Dropout(d_rate))
            self.layers.add_module("fc3", torch.nn.Linear(dim3, 1))
        else:
 #           self.layers.add_module("drop_out", torch.nn.Dropout(d_rate))
            self.layers.add_module("fc2", torch.nn.Linear(dim2, 1))
        
        # 因为score的分数是从-1到1， 所以对应的结果是否加一个激活函数会比较好
        if act_func_out:
            self.layers.add_module(act_func_out, getattr(torch.nn, act_func_out)())

    def forward(self, input):
        """
        input: (batch_size, 1000)
        output: (1)
        """
        out = self.layers(input)
        return out

class BiLinear(torch.nn.Module):
    def __init__(self, dim2 = None, act_func = 'Tanh', act_func_out = None):
        super(BiLinear, self).__init__()
        dim1 = 500
        self.li_sys = torch.nn.Linear(dim1, dim1, bias = False)
        self.li_ref = torch.nn.Linear(dim1, dim1, bias = False)
        self.act_func = getattr(torch.nn, act_func)()
        self.fc = None
        if dim2:
            self.fc = torch.nn.Linear(dim1, dim2)
            self.drop_out = torch.nn.Dropout(0.5)
            self.li_out = torch.nn.Linear(dim2, 1)
        else:
            self.li_out = torch.nn.Linear(dim1, 1)
        self.act_func_out = None
        if act_func_out:
            self.act_func_out = getattr(torch.nn, act_func_out)()
    
    def forward(self, input):
        """
        input:
            input: (batch, 1000)
        output:
            score: (1)
        """
        input_sys = input[:,:500]
        input_ref = input[:,500:]
        proj_sys = self.li_sys(input_sys)
        proj_ref = self.li_ref(input_ref)
        sum_in = proj_sys + proj_ref
        acted_sum_in = self.act_func(sum_in)
        if self.fc:
            acted_sum_in = self.drop_out(self.fc(acted_sum_in))
        out = self.li_out(acted_sum_in)
        if self.act_func_out:
            out = self.act_func_out(out)
        return out
