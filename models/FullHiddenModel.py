import torch
import torch.nn as nn
import numpy
import math

from Attention import ScaledDotProductAttention, MultiHeadAttention

class MlpModel:
    pass

class MultiHeadAttnMlpModel(nn.Module):
    def __init__(self, num_head = 8, num_dim_k = 64, num_dim_v = 64, d_rate_attn = 0.1, act_func1 = "LeakyReLU", dim2 = 100, act_func2 = "LeakyReLU"):
        """
        num_head: for Attn, the number of head in MultiHeadAttention
        num_dim_k: for Attn, the number of dimension query and key will mapping to
        num_dim_v: for Attn, the number of dimension value will mapping to
        d_rate_attn: drop out rate for MultiHeadAttention 
        """
        super(MultiHeadAttnMlpModel, self).__init__()
        num_dim = 500
        num_seq = 100
        self.attn =  MultiHeadAttention(num_head, num_dim, num_dim_k, num_dim_v, d_rate_attn)
        self.bn = nn.BatchNorm1d(num_dim)
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(num_seq*num_dim, num_dim))
        self.mlp.add_module('bn1', nn.BatchNorm1d(num_dim))
        self.mlp.add_module('act_fun1', getattr(nn, act_func1)())
        self.mlp.add_module('fc2', nn.Linear(num_dim, dim2))
        self.mlp.add_module('bn2', nn.BatchNorm1d(dim2))
        self.mlp.add_module('act_fun2', getattr(nn, act_func2)())
        self.mlp.add_module('fc3', nn.Linear(dim2, 1))

    def forward(self, data_in):
        """
        data_in: (batch, seq_len * 2, num_dim)
        """
        seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_ref, data_in_ref, data_in_ref)
        batch_size, num_q, num_dim = data_attn.size()
        data_attn = data_attn.view(batch_size, -1)
        out = self.mlp(data_attn)
        return out

class MultiHeadAttnLSTMModel(nn.Module):
    def __init__(self, num_head = 8, num_dim_k = 64, num_dim_v = 64, d_rate_attn = 0.1, dim2 = 100, act_func2 = "LeakyReLU"):
        num_dim  = 500
        super(MultiHeadAttnLSTMModel, self).__init__()
        self.attn = MultiHeadAttention(num_head, num_dim, num_dim_k, num_dim_v, d_rate_attn)
        self.rnn = nn.LSTM(input_size = 500, hidden_size = 500, num_layers = 2)
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(500, dim2))
        self.mlp.add_module('bn2', nn.BatchNorm1d(dim2))
        self.mlp.add_module('act_fun2', getattr(nn, act_func2)())
        self.mlp.add_module('fc3', nn.Linear(dim2, 1))

    def forward(self, data_in):
        seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_ref, data_in_ref, data_in_ref)
        # ? the input of the rnn is (batch_size, seq_len, num_dim), but the return of the rnn is
        # ? (batch_size, seq_len, hidden_size), nach the document, i should got the (seq_len, batch_size, hidden_size). Why??
        # data_rnn: (seq_len, batch_size, hidden_size)
        data_rnn, _ = self.rnn(data_attn)
        #data_rnn = torch.transpose(data_rnn, 0, 1)
        lengths_ref = getSentenceLengths(data_in_ref)
        data_selected = index_select(data_rnn, lengths_ref)
        out = self.mlp(data_selected)
        return out

class MultiHeadAttnConvModel(nn.Module):
    def __init__(self, num_head = 8, num_dim_k = 64, num_dim_v = 64, d_rate_attn = 0.1, dim1 = 20, act_func1 = "LeakyReLU", kernel_size1 = 3, stride1 = 2, kernel_size2 = 3, stride2 = 2):
        num_dim = 500
        seq_len = 100
        super(MultiHeadAttnConvModel, self).__init__()
        self.attn = MultiHeadAttention(num_head, num_dim, num_dim_k, num_dim_v, d_rate_attn)
        self.layers = nn.Sequential()
        self.layers.add_module("conv1", nn.Conv1d(seq_len, dim1, kernel_size1, stride1))
        self.layers.add_module("bn1", nn.BatchNorm1d(dim1))
        self.layers.add_module("act_fun1", getattr(nn, act_func1)())
        self.layers.add_module("conv2", nn.Conv1d(dim1, 1, kernel_size2, stride2))
        self.layers.add_module("maxpool", nn.MaxPool1d(124))

    def forward(self, data_in):
        seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim=1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_ref, data_in_ref, data_in_ref)
        out = self.layers(data_attn)
        return out

def index_select(src, indexes):
    """
    in:
        src: (batch_size, seq_len, num_dim)
        index: list 
    out: (batch_size, num_dim)
    """
    assert(len(src)==len(indexes))
    out = []
    for index, item in enumerate(src):
        out.append(item[indexes[index],:])
    out = torch.stack(out, 1)
    out = torch.transpose(out, 0, 1)
    return out

def getSentenceLengths(data_in):
    """
    data_in: (batch_size, seq_len, num_dim)
    out: list of length
    """
    lengths = []
    for item in data_in:
        counter = 0
        for sub_item in item:
            if int(int(math.ceil(torch.sum(torch.abs(sub_item.data))))) == 0:
                break;
            counter = counter + 1
        if counter >= 100:
            counter = 100
        lengths.append(counter - 1) 
    return lengths
