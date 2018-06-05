import sys
sys.path.append('../utils')
import torch
import torch.nn as nn
import numpy
import math
import nnActi

from Attention import ScaledDotProductAttention, MultiHeadAttention

seq_len = 100
class Conv2dMlpModel_rank(nn.Module):
    def __init__(self, dim2 = 16, act_func = "LeakyReLU", softmax = True):
        super(Conv2dMlpModel_rank, self).__init__()
        dim_w = 300
        dim_h = 30
        # conv_layer1 for one word feature
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module('conv1', nn.Conv2d(3, 16, (2,500), stride = (1,500)))
        self.conv_layer.add_module('dp1', nn.Dropout2d(0.1))
        self.conv_layer.add_module('af1', nnActi.get_acti('ReLU'))
        #self.conv_layer.add_module('mp1', nn.MaxPool2d((1,1)))
        # conv_layer2 for two words feature
        self.conv_layer2 = nn.Sequential()
        self.conv_layer2.add_module('conv2', nn.Conv2d(3, 16, (2,500), stride = (1,500), padding = (1,0)))
        self.conv_layer2.add_module('dp2', nn.Dropout2d(0.1))
        self.conv_layer2.add_module('af2', nnActi.get_acti('ReLU'))
        #self.conv_layer.add_module('mp2', nn.MaxPool2d((2,1)))
        # conv_layer3 for three words feature
        self.conv_layer3 = nn.Sequential()
        self.conv_layer3.add_module('conv3', nn.Conv2d(3, 16, (3,500), stride = (1,500), padding = (1,0)))
        self.conv_layer3.add_module('dp3', nn.Dropout2d(0.1))
        self.conv_layer3.add_module('af3', nnActi.get_acti('ReLU'))
        #self.conv_layer.add_module('mp3', nn.MaxPool2d((2,5)))
        self.mlp = nn.Sequential()
        self.mlp.add_module('li1', nn.Linear(480, 16))
        if softmax:
            self.mlp.add_module('li2', nn.Linear(16, 3))
        else:
            self.mlp.add_module('li2', nn.Linear(16, 3))

    def forward(self, data_in):
        seq_len = 10
        shapes = data_in.data.shape
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_s1 = data_in_chunks[0].unsqueeze(dim = 1)
        data_in_s2 = data_in_chunks[1].unsqueeze(dim = 1)
        data_in_ref = data_in_chunks[2].unsqueeze(dim = 1)
        # shape of conv2d input is: (batch_size, channels, height, width)
        data_in = torch.cat((data_in_s1, data_in_s2, data_in_ref), 1)
        num_batch = shapes[0]
        out_conv1 = self.conv_layer(data_in).view(num_batch, -1)
        out_conv2 = self.conv_layer2(data_in).view(num_batch, -1)
        out_conv3 = self.conv_layer3(data_in).view(num_batch, -1)
        out_conv = torch.cat((out_conv1,out_conv2, out_conv3), 1)
        out = self.mlp(out_conv)
        return out

class Conv3dMlpModel_rank(nn.Module):
    def __init__(self, dim2 = 15, act_func = "LeakyReLU", softmax = True):
        super(Conv3MlpModel_rank, self).__init__()
        dim_w = 300
        dim_h = 30
        self.conv_layer = nn.Sequential()
        self.conv_layer.add_module('conv1', nn.Conv3d(1,128, (1,2)))
        self.conv_layer.add_module('np1', nn.Dropout3d(0.1))
        self.conv_layer.add_module('af1', nnActi.get_acti('ReLU'))
        self.conv_layer.add_module('mp1', nn.MaxPool3d((1,1,2)))
        self.conv_layer.add_module('conv2', nn.Conv3d(128, 16, (2,2)))
        self.conv_layer.add_module('dp2', nn.Dropout3d(0.1))
        self.conv_layer.add_module('af2', nnActi.get_acti('ReLU'))
        self.conv_layer.add_module('mp2', nn.MaxPool3d((1,2,2)))
        self.conv_layer.add_module('conv3', nn.Conv3d(16, 1, (5,5)))
        self.conv_layer.add_module('dp3', nn.Dropout3d(0.1))
        self.conv_layer.add_module('af3', nnActi.get_acti('ReLU'))
        self.conv_layer.add_module('mp3', nn.MaxPool3d((1,5,5)))
        if softmax:
            self.mlp = nn.Linear(28*3, 3)
        else:
            self.mlp = nn.Linear(28*3, 3)


    def forward(self, data_in):
        seq_len = 10
        shapes = data_in.data.shape
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_s1 = data_in_chunks[0].unsqueeze(dim = 1).unsqueeze(dim = 1)
        data_in_s2 = data_in_chunks[1].unsqueeze(dim = 1).unsqueeze(dim = 1)
        data_in_ref = data_in_chunks[2].unsqueeze(dim = 1).unsqueeze(dim = 1)
        # shape of conv3d input is: (batch_size, channels, depth, height, width)
        data_in = torch.cat((data_in_s1, data_in_s2, data_in_ref), 2)
        out_conv = self.conv_layer(data_in)
        out_conv = out_conv.view(shapes[0], -1)
        out = self.mlp(out_conv)
        return out

class LSTMMlpModel_rank(nn.Module):
    def __init__(self, dim2 = 128, dim3 = 16, act_func = "LeakyReLU", softmax = True):
        num_dim = 300
        super(LSTMMlpModel_rank, self).__init__()
        self.rnn_ref = nn.LSTM(input_size = num_dim, hidden_size = num_dim, num_layers = 1)
        self.rnn_s1 = nn.LSTM(input_size = num_dim, hidden_size = num_dim, num_layers = 1)
        self.rnn_s2 = nn.LSTM(input_size = num_dim, hidden_size = num_dim, num_layers = 1)
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(num_dim*3, dim2))
        self.mlp.add_module('bn1', nn.BatchNorm1d(dim2))
        self.mlp.add_module('act_fun', nnActi.get_acti(act_func))
        self.mlp.add_module('fc2', nn.Linear(dim2, dim3))
        self.mlp.add_module('bn2', nn.BatchNorm1d(dim3))
        self.mlp.add_module('act_fun2', nnActi.get_acti(act_func))
        if softmax:
            self.mlp.add_module('fc3', nn.Linear(dim3, 3))
        else:
            self.mlp.add_module('fc3', nn.Linear(dim3, 1))

    def forward(self, data_in):
        seq_len = 30
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_s1 = data_in_chunks[0]
        data_in_s2 = data_in_chunks[1]
        data_in_ref = data_in_chunks[2]
        # input of rnn: (seq_len, batch, input_size)
        data_in_s1 = torch.transpose(data_in_s1, 0, 1)
        data_in_s2 = torch.transpose(data_in_s2, 0, 1)
        data_in_ref = torch.transpose(data_in_ref, 0, 1)
        # run rnn, rnn has two output 
        out_rnn_ref, _ = self.rnn_ref(data_in_ref)
        out_rnn_s1, _ = self.rnn_s1(data_in_s1)
        out_rnn_s2, _  = self.rnn_s2(data_in_s2)
        # output of rnn: (seq_len, batch, input_size)
        out_rnn_ref = torch.transpose(out_rnn_ref, 0, 1)[:,-1,:]
        out_rnn_s1 = torch.transpose(out_rnn_s1, 0, 1)[:,-1,:]
        out_rnn_s2 = torch.transpose(out_rnn_s2, 0, 1)[:,-1,:]
        # cat the data and run mlp
        out_rnn = torch.cat((out_rnn_s1, out_rnn_s2, out_rnn_ref),1)
        out = self.mlp(out_rnn)
        return out

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
        self.mlp.add_module('act_fun1', nnActi.get_acti(act_func1))
        self.mlp.add_module('fc2', nn.Linear(num_dim, dim2))
        self.mlp.add_module('bn2', nn.BatchNorm1d(dim2))
        self.mlp.add_module('act_fun2', nnActi.get_acti(act_func2))
        self.mlp.add_module('fc3', nn.Linear(dim2, 1))

    def forward(self, data_in):
        """
        data_in: (batch, seq_len * 2, num_dim)
        """
        #seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_ref, data_in_sys, data_in_sys)
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
        self.mlp.add_module('act_fun2', nnActi.get_acti(act_func2))
        self.mlp.add_module('fc3', nn.Linear(dim2, 1))

    def forward(self, data_in):
        #seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_ref, data_in_sys, data_in_sys)
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
    def __init__(self, num_head = 8, num_dim_k = 64, num_dim_v = 64, d_rate_attn = 0.1, dim1 = 20, act_func1 = "LeakyReLU", kernel_size1 = 3, stride1 = 2, act_func2 = "LeakyReLU", kernel_size2 = 3, stride2 = 2):
        """
        problematic!!!
        what is the dimension in col direction??
        """
        num_dim = 500
        #seq_len = 100
        super(MultiHeadAttnConvModel, self).__init__()
        self.attn = MultiHeadAttention(num_head, num_dim, num_dim_k, num_dim_v, d_rate_attn)
        self.dim_conv_out1 = get_dim_out(seq_len, kernel_size1, stride1)
        self.dim_conv_out2 = get_dim_out(self.dim_conv_out1, kernel_size2, stride2)
        self.layers = nn.Sequential()
        self.layers.add_module("conv1", nn.Conv1d(num_dim, dim1, kernel_size1, stride1))
        self.layers.add_module("bn1", nn.BatchNorm1d(dim1))
        self.layers.add_module("act_func1", nnActi.get_acti(act_func1))
        if self.dim_conv_out2 < 1:
            self.layers.add_module("conv2", nn.Conv1d(dim1, 1, 2, 1))
            self.dim_conv_out = get_dim_out(self.dim_conv_out1, 2, 1)
        else:
            self.layers.add_module("conv2", nn.Conv1d(dim1, 1, kernel_size2, stride2))
            self.dim_conv_out = self.dim_conv_out2
        self.layers.add_module('bn2', nn.BatchNorm1d(1))
        self.layers.add_module('act_func2', nnActi.get_acti(act_func2))
        #self.layers.add_module("maxpool", nn.MaxPool1d(124))
        self.li = nn.Linear(self.dim_conv_out, 1, bias = True)

    def forward(self, data_in):
        #seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim=1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_ref, data_in_sys, data_in_sys)
        data_attn = data_attn.transpose(1,2)
        #print data_attn.size()
        data_conv = self.layers(data_attn)
        #print data_conv.size()
        data_conv = data_conv.squeeze()
        if self.dim_conv_out == 1:
            data_conv = data_conv.unsqueeze(1)
            #print data_conv.size()
        out = self.li(data_conv)
        #print out.size()
        return out

class MultiHeadAttnConvModel2(nn.Module):
    def __init__(self, num_head = 8, num_dim_k = 64, num_dim_v = 64, d_rate_attn = 0.1, dim1 = 20, act_func1 = "LeakyReLU", kernel_size1 = 3, stride1 = 2, act_func2 = "LeakyReLU", kernel_size2 = 3, stride2 = 2):
        """
        problematic
        same problem as described above
        """
        num_dim = 500
        #seq_len = 100
        super(MultiHeadAttnConvModel2, self).__init__()
        self.attn = MultiHeadAttention(num_head, num_dim, num_dim_k, num_dim_v, d_rate_attn)
        self.dim_conv_out1 = get_dim_out(seq_len, kernel_size1, stride1)
        self.dim_conv_out2 = get_dim_out(self.dim_conv_out1, kernel_size2, stride2)
        self.layers = nn.Sequential()
        self.layers.add_module("conv1", nn.Conv1d(num_dim, dim1, kernel_size1, stride1))
        self.layers.add_module("bn1", nn.BatchNorm1d(dim1))
        self.layers.add_module("act_func1", nnActi.get_acti(act_func1))
        if self.dim_conv_out2 < 1:
            self.layers.add_module("conv2", nn.Conv1d(dim1, 1, 2, 1))
            self.dim_conv_out = get_dim_out(self.dim_conv_out1, 2, 1)
        else:
            self.layers.add_module("conv2", nn.Conv1d(dim1, 1, kernel_size2, stride2))
            self.dim_conv_out = self.dim_conv_out2
        self.layers.add_module('bn2', nn.BatchNorm1d(1))
        self.layers.add_module('act_func2', nnActi.get_acti(act_func2))
        #self.layers.add_module("maxpool", nn.MaxPool1d(124))
        self.li = nn.Linear(self.dim_conv_out, 1, bias = True)

    def forward(self, data_in):
        #seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim=1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_sys, data_in_ref, data_in_ref)
        data_attn = data_attn.transpose(1,2)
        #print data_attn.size()
        data_conv = self.layers(data_attn)
        #print data_conv.size()
        data_conv = data_conv.squeeze()
        if self.dim_conv_out == 1:
            data_conv = data_conv.unsqueeze(1)
            #print data_conv.size()
        out = self.li(data_conv)
        #print out.size()
        return out

class ScaledDotAttnConvModel(nn.Module):
    def __init__(self, d_rate_attn = 0.1, dim1 = 20, act_func1 = "LeakyReLU", kernel_size1 = 3, stride1 = 2, act_func2 = "LeakyReLU", kernel_size2 = 3, stride2 = 2):
        """
        problematic
        same proble as described above
        """
        num_dim = 500
        #seq_len = 100
        super(ScaledDotAttnConvModel, self).__init__()
        self.attn = ScaledDotProductAttention(num_dim, d_rate_attn)
        self.dim_conv_out1 = get_dim_out(seq_len, kernel_size1, stride1)
        self.dim_conv_out2 = get_dim_out(self.dim_conv_out1, kernel_size2, stride2)
        self.layers = nn.Sequential()
        self.layers.add_module("conv1", nn.Conv1d(num_dim, dim1, kernel_size1, stride1))
        self.layers.add_module("bn1", nn.BatchNorm1d(dim1))
        self.layers.add_module("act_func1", nnActi.get_acti(act_func1))
        if self.dim_conv_out2 < 1:
            self.layers.add_module("conv2", nn.Conv1d(dim1, 1, 2, 1))
            self.dim_conv_out = get_dim_out(self.dim_conv_out1, 2, 1)
        else:
            self.layers.add_module("conv2", nn.Conv1d(dim1, 1, kernel_size2, stride2))
            self.dim_conv_out = self.dim_conv_out2
        self.layers.add_module('bn2', nn.BatchNorm1d(1))
        self.layers.add_module('act_func2', nnActi.get_acti(act_func2))
        #self.layers.add_module("maxpool", nn.MaxPool1d(124))
        self.li = nn.Linear(self.dim_conv_out, 1, bias = True)

    def forward(self, data_in):
        #seq_len = 100
        data_in_chunks = torch.split(data_in, seq_len, dim=1)
        data_in_sys = data_in_chunks[0]
        data_in_ref = data_in_chunks[1]
        data_attn, _ = self.attn(data_in_ref, data_in_sys, data_in_sys)
        data_attn = data_attn.transpose(1,2)
        #print data_attn.size()
        data_conv = self.layers(data_attn)
        #print data_conv.size()
        data_conv = data_conv.squeeze()
        if self.dim_conv_out == 1:
            data_conv = data_conv.unsqueeze(1)
            #print data_conv.size()
        out = self.li(data_conv)
        #print out.size()
        return out

class ScaledDotAttnMlpModel_rank(nn.Module):
    def __init__(self, d_rate_attn = 0.1, dim2 = 256, dim3 = 64, act_func = "LeakyReLU", softmax = True):
        num_dim = 500
        super(ScaledDotAttnMlpModel_rank, self).__init__()
        self.attn_s1 = ScaledDotProductAttention(num_dim, d_rate_attn)
        self.attn_s2 = ScaledDotProductAttention(num_dim, d_rate_attn)
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(num_dim*2, dim2))
        self.mlp.add_module('bn1', nn.BatchNorm1d(dim2))
        self.mlp.add_module('act_fun', nnActi.get_acti(act_func))
#        self.mlp.add_module('fc2', nn.Linear(dim2, dim3))
#        self.mlp.add_module('bn2', nn.BatchNorm1d(dim3))
#        self.mlp.add_module('act_fun2', nnActi.get_acti(act_func))
        if softmax:
            self.mlp.add_module('fc3', nn.Linear(dim2, 3))
        else:
            self.mlp.add_module('fc3', nn.Linear(dim2, 1))

    def forward(self, data_in):
        #seq_len = 30
        data_in_chunks = torch.split(data_in, seq_len, dim = 1)
        data_in_s1 = data_in_chunks[0]
        data_in_s2 = data_in_chunks[1]
        data_in_ref = data_in_chunks[2]
        # output of Attn: (batch, num_q, num_dim)
        out_attn_s1, _ = self.attn_s1(data_in_ref, data_in_s1, data_in_s1)
        out_attn_s2, _ = self.attn_s2(data_in_ref, data_in_s2, data_in_s2)
        # get sum of words
        out_attn_s1 = torch.sum(out_attn_s1, dim = 1)
        out_attn_s2 = torch.sum(out_attn_s2, dim = 1)
        out_attn = torch.cat((out_attn_s1, out_attn_s2), 1)
        out = self.mlp(out_attn)
        return out




######################
# assist functions
######################
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

def get_dim_out(dim_in, kernel_size, stride, padding = 0, dilation = 1):
    """
    calculate number of the output dimention for the convolutional network
    """
    dim_out =  int(math.floor((dim_in + 2 * padding - dilation * (kernel_size -1) - 1)/stride + 1))
    #print dim_out
    return dim_out
