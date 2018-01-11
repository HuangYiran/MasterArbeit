import torch
import torch.nn as nn
import numpy

from Attention import MultiHeadAttention

class MlpModel:
    pass

class MultiHeadAttnMlpModel(nn.Module):
    def __init__(self, num_head, num_dim_k, num_dim_v, d_rate_attn = 0.1, act_func1 = "LeakyReLU", dim2 = 100, act_func2 = "LeakyReLU"):
        """
        num_head: for Attn, the number of head in MultiHeadAttention
        num_dim_k: for Attn, the number of dimension query and key will mapping to
        num_dim_v: for Attn, the number of dimension value will mapping to
        d_rate_attn: drop out rate for MultiHeadAttention 
        """
        super(MultiHeadAttenMlpModel, self).__init__()
        num_dim = 500
        num_seq = 100
        self.attn =  MultiHeadAttention(num_head, num_dim, num_dim_k, num_dim_v, d_rate_attn))
        self.bn = nn.BatchNorm1d(num_dim)
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(num_seq*num_dim, num_dim))
        self.mlp.add_module('bn1', nn.BatchNorm1d(num_dim))
        self.mlp.add_module('act_fun1', getattr(nn, act_func1)())
        self.mlp.add_module('fc2', nn.Linear(num_dim, dim2))
        self.mlp.add_module('bn2', nn.BatchNorm1d(dim2))
        self.mlp.add_module('act_fun2', getattr(nn, act_func2)())
        self.mlp.add_module('fc3', nn.Linear(dim2, 1))

    def vorward(self, data_in):
        """
        data_in: (batch, seq_len * 2, num_dim)
        """
        seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn = self.attn(data_in_ref, data_in_ref, data_in_ref)
        batch_size, num_q, num_dim = data_attn.size()
        data_attn = data_attn.view(batch_size, -1)
        out = self.layers(data_attn)
        return out
