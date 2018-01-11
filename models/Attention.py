import torch
import torch.nn as nn
import numpy
import sys

from Modules import Linear

class ScaledDotProductAttention(nn.Module):
    def __init__(self, num_dim, dropout_rate = 0.1):
        """
        num_dim: the number of dimension of each query word
        query word and key should have the same num_dim
        number of dimension of value vector can be different
        """
        super(scaledDotProductAtttention, self).__init__()
        self.scala = numpy.power(num_dim, 0.5)
        self.dropout = nn.Dorpout(dropout_rate)
        self.softmax = nn.softmax()

    def forward(self, query, key ,value):
        """
        input:
            query: (batch_size, num_q, num_dim)
            key: (batch_size, num_v, num_dim)
            value: (batch_size, num_v, num_dim_value)
        output:
            out: (batch_size, num_q, num_dim_value)
            attn: (batch_size, num_q, num_v)
        """
        # attn: (batch_size, num_q, num_v)
        attn = torch.bmm(query, key.transport(1,2))/self.scala
        attn = self.dropout(attn)
        attn = self.softmax(attn)
        # out: (batch_size, num_q, num_dim)
        out = torch.bmm(attn, value)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, num_dim, num_dim_k, num_dim_v, dropout_rate = 0.1):
        """
        num_head: the number of head
        num_dim: the number of dimension of each query word and key
        num_dim_k: the number of dimension query and key will mapping to 
        num_dim_v: the number of dimension value will mapping to 
        """
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.num_dim = num_dim
        self.num_dim_k = num_dim_k
        self.num_dim_v = num_dim_v

        # parameter w_q, w_v, w_k for all head 
        self.w_q = torch.Parameter(torch.FloatTensor(num_head, num_dim, num_dim_k))
        self.w_k = torch.Parameter(torch.FloatTensor(num_head, num_dim, num_dim_k))
        self.w_v = torch.Parameter(torch.FloatTensor(num_head, num_dim, num_dim_v))
        nn.init_xavier_normal(self.w_q)
        nn.init_xavier_normal(self.w_k)
        nn.init_xavier_normal(self.w_k)

        self.attention = ScaledDotProductAttention(num_dim)
        self.project = Linear(num_head*num_dim_v, num_dim)

        self.dropout = nn.dropout(dropout_rate)

    def forward(self, query, key, value):
        """
        input:
            query: (batch_size, num_q, num_dim)
            key: (batch_size, num_v, num_dim)
            value: (batch_size, num_v, num_dim)
        output:
            out: (batch_size, num_q, num_dim)
            attns: (num_head*batch_size, num_q, num_v)
        """
        # (batch_size, num_q, num_dim)->(batch_size*num_head, num_q, num_dim)->(num_head, batch_size*num_q, num_dim)
        queries = query.repeat(num_head, 1, 1).view(num_head, -1, num_dim)
        keys = key.repeat(num_head, 1, 1).view(num_head, -1, num_dim)
        values = value.repeat(num_head, 1, 1).view(num_head, -1, num_dim)

        num_q = query.size()[1]
        num_k = key.size()[1]
        num_v = value.size()[1]
        # bmm->(num_head, batch_size*num_q, num_k) -> (num_head*batch_size, num_q, num_k)
        queries = torch.bmm(queries, self.w_q).view(-1, num_q, num_dim_k)
        keys = torch.bmm(keys, self.w_k).view(-1, num_k, num_dim_k)
        values = torch.bmm(keys, self.w_k).view(-1, num_v, num_dim_v)
        # outs: (num_head*batch_size, num_q, num_dim_v), attns: (num_head*batch_size, num_q, num_v)
        outs, attns = self.attention(queries, keys, values)

        batch_size = query.size()[0]
        # outs: (batch_size, num_q, num_head * num_dim_v)
        outs = torch.cat(torch.split(outs, batch_size, dim = 0), dim = -1)

        # outs: (batch_size, num_q, num_dim)
        outs = self.proj(outs)
        outs = self.dropout(outs)

        return outs, attns
