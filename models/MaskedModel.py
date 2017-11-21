# -*- coding:UTF-8 -*-
import torch
import numpy

"""
    基本思想和embedding的多维含义显现相似。
    认为并不是vector中，每个维度的信息都是对评分有用的，所以中心思想是通过mask屏蔽掉一些维度的信息
    核心问题是如和得到这个mask，哪些信息应该作为计算的主要信息
"""
class MaskedModel1(torch.nn.Module):
    """
    这是使用ref信息来生成mask，把结果用于sys信息上。
    """
    def __init__(self, dim2 = 10, act_func = 'Tanh'):
        super(MaskedModel1, self).__init__()
        self.li_1 = torch.nn.Linear(500, dim2)
        self.li_out = torch.nn.Linear(dim2, 1)
        self.li_mask = torch.nn.Linear(500, 500)
        self.sf = torch.nn.Softmax()
        self.act = getattr(torch.nn, act_func)()

    def forward(self, input):
        """
        input:
            input: (batch, 1000)
        output:
            score: (1)
        """
        input_sys = input[:, :500]
        input_ref = input[:, 500:]
        proj_ref = self.li_mask(input_ref)
        mask = self.sf(proj_ref)
        masked_sys = input_sys * mask
        proj_masked_sys = self.li_1(masked_sys)
        proj_masked_sys = self.act(proj_masked_sys)
        out = self.li_out (proj_masked_sys)
        return out

class MaskedModel2(torch.nn.Module):
    """
    使用综合信息作为mask源，同时把结果作用于sys
    相当于用综合信息给sys中各个维度的信息进行评分
    """
    def __init__(self, dim2 = 20, act_func = 'Tanh'):
        super(MaskedModel2, self).__init__()
        self.li_sys = torch.nn.Linear(500, 500, bias = False)
        self.li_ref = torch.nn.Linear(500, 500, bias = False)
        self.li_mask = torch.nn.Linear(500, 500)
        self.sf = torch.nn.Softmax()
        #self.li_1 = torch.nn.Linear(500, dim2)
        self.li_out = torch.nn.Linear(500, 1)
        self.act_func = getattr(torch.nn, act_func)()

    def forward(self, input):
        """
        input:
            input (batch_size, 1000)
        output:
            score (1)
        """
        input_sys = input[:,:500]
        input_ref = input[:,500:]
        proj_sys = self.li_sys(input_sys)
        proj_ref = self.li_ref(input_ref)
        sum_in = proj_sys + proj_ref
        mask = self.li_mask(sum_in)

        masked_sys = mask * input_sys
        out = self.li_out(masked_sys)
#        proj_masked_sys = self.act_func(self.li_1(masked_sys))
#        out = self.li_out(proj_masked_sys)
        return out
