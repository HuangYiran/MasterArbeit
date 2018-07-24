# -*- coding: UTF-8 -*-
import sys
sys.path.append('../utils/')
sys.path.append('./models/')
import torch
import numpy
import nnActi

from Layers import CapsLayer
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, mahalanobis, minkowski, seuclidean, sqeuclidean, wminkowski

class TwoLayerLinear(torch.nn.Module):
    def __init__(self, act_func = "ReLU", mom = 0.1):
        super(TwoLayerLinear, self).__init__()
        dim1 = 1000
        self.layers = torch.nn.Sequential()
        self.layers.add_module('fc1', torch.nn.Linear(dim1, 1)) # 整个网络就一个神经元，输入是1000，输出使1
        self.layers.add_module('bn', torch.nn.BatchNorm1d(1))
        self.layers.add_module('act', nnActi.get_acti(act_func))

    def forward(self, input):
        """
        input: (batch_size, 1000)
        output: (1)
        """
        out = self.layers(input)
        return out



class BasicLinear(torch.nn.Module):
    """
    multi-layer linear model: 1000 - dim2 - dim3 - 1 
    use batch normalization, problem is that each unit needs two more parameters. 
    ?? should i add a bn for the firt layer for each batch input, other direct normalize all the input 
    ?? because i have done the BN. Does that means i can delete the bias in linear model if so i can save some parameter.
    ?? drop out has the function of Ensemble can it improve the result
    """
    def __init__(self, dim2 = 500, dim3 = None, act_func = "ReLU", act_func_out = None, mom = 0.1, num_dim = 500):
        super(BasicLinear, self).__init__()
        self.num_dim = num_dim
        dim1 = self.num_dim*2
        self.layers = torch.nn.Sequential()
        #self.layers.add_module("bn0", torch.nn.BatchNorm1d(dim1))
        #self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2, bias = False))
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        self.layers.add_module("bn", torch.nn.BatchNorm1d(dim2))
        self.layers.add_module(act_func + "1", nnActi.get_acti(act_func))
 
        if dim3:
            #self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3, bias = False))
            self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3))
            self.layers.add_module("bn2", torch.nn.BatchNorm1d(dim3, momentum = mom))
            self.layers.add_module(act_func + "2", nnActi.get_acti(act_func))
            self.layers.add_module("drop_out", torch.nn.Dropout(0.5))
            self.layers.add_module("fc3", torch.nn.Linear(dim3, 1))
        else:
            self.layers.add_module("drop_out", torch.nn.Dropout(0.5))
            self.layers.add_module("fc2", torch.nn.Linear(dim2, 1))
        
        # 因为score的分数是从-1到1， 所以对应的结果是否加一个激活函数会比较好
        if act_func_out:
            self.layers.add_module("bn_out",  torch.nn.BatchNorm1d(dim3, momentum = mom))
            self.layers.add_module(act_func_out, nnActi.get_acti(act_func_out))

    def forward(self, input):
        """
        input: (batch_size, 1000)
        output: (1)
        """
        out = self.layers(input)
        return out

class BasicLinear_dropout(torch.nn.Module):
    """
    multi-layer linear model: 1000 - dim2 - dim3 - 1 
    only use drop out 
    i am not sure if it can convergence quickly without BN. 
    try to normalize the input for the training. 
    only use the activation function without core area.
    use drop out to get a besser result.
    """
    def __init__(self, dim2 = 500, dim3 = None, act_func = "ReLU", act_func_out = None, d_rate1 = 0.5, d_rate2 = 0.5):
        super(BasicLinear_dropout, self).__init__()
        dim1 = 1000
        self.layers = torch.nn.Sequential()
        self.layers.add_module("fc1", torch.nn.Linear(dim1, dim2))
        self.layers.add_module(act_func + "1", nnActi.get_acti(act_func))
 
        if dim3:
            self.layers.add_module("fc2", torch.nn.Linear(dim2, dim3))
            self.layers.add_module(act_func + "2", nnActi.get_acti(act_func))
            self.layers.add_module("drop_out", torch.nn.Dropout(d_rate1))
            self.layers.add_module("fc3", torch.nn.Linear(dim3, 1))
        else:
            self.layers.add_module("drop_out", torch.nn.Dropout(d_rate2))
            self.layers.add_module("fc2", torch.nn.Linear(dim2, 1))
        
        # 因为score的分数是从-1到1， 所以对应的结果是否加一个激活函数会比较好
        if act_func_out:
            self.layers.add_module(act_func_out, nnActi.get_acti(act_func_out))

    def forward(self, input):
        """
        input: (batch_size, 1000)
        output: (1)
        """
        out = self.layers(input)
        return out

class BiLinear(torch.nn.Module):
    def __init__(self, dim2 = 64, dim3 = None, act_func = 'Tanh', act_func_out = None, d_rate = 0.5):
        super(BiLinear, self).__init__()
        dim1 = 500
        self.li_sys = torch.nn.Linear(dim1, dim1, bias = False)
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
            self.layers.add_module("fc_out", torch.nn.Linear(dim3, 1))
        else:
            self.layers.add_module("fc_out", torch.nn.Linear(dim2, 1))

        if act_func_out:
            self.layers.add_module("act_out", nnActi.get_acti(act_func_out))

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
        out = self.layers(acted_sum_in)
        return out

class Simple0(torch.nn.Module):
    def __init__(self):
        super(Simple0, self).__init__()
        dim2 = 500
        act_func = 'Tanh'
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(1000, dim2))
        self.mlp.add_module('act_func1', nnActi.get_acti(act_func))
        self.mlp.add_module('dp1', torch.nn.Dropout(0.5))
        self.mlp.add_module('fc2', torch.nn.Linear(dim2, 1))
        self.li = torch.nn.Linear(2,3)
        self.sf = torch.nn.LogSoftmax()

    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        inp1 = torch.cat([s1, ref], 1)
        inp2 = torch.cat([s2, ref], 1)
        out1 = self.mlp(inp1)
        out2 = self.mlp(inp2)
        out = self.sf(self.li(torch.cat([out1, out2], 1)))
        return out

class Simple1(torch.nn.Module):
    def __init__(self):
        super(Simple1, self).__init__()
        dim2 = 500
        act_func = 'Tanh'
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(1000, dim2))
        self.mlp.add_module('act_func1', nnActi.get_acti(act_func))
        self.mlp.add_module('dp1', torch.nn.Dropout(0.5))
        self.mlp.add_module('fc2', torch.nn.Linear(dim2, 1))

    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        inp1 = torch.cat([s1, ref], 1)
        inp2 = torch.cat([s2, ref], 1)
        out1 = self.mlp(inp1)
        out2 = self.mlp(inp2)
        out = out1-out2
        out = out.squeeze()
        return out

class Simple2(torch.nn.Module):
    """
    mul input
    """
    def __init__(self):
        super(Simple2, self).__init__()
        dim2 = 500
        act_func = 'ReLU'
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(500, dim2))
        self.mlp.add_module('act_func1', nnActi.get_acti(act_func))
#        self.mlp.add_module('bn1', torch.nn.BatchNorm1d(dim2))
        self.mlp.add_module('dp1', torch.nn.Dropout(0.5))
        self.mlp.add_module('fc2', torch.nn.Linear(dim2, 1))

    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        #inp1 = torch.cat([s1, ref], 1)
        #inp2 = torch.cat([s2, ref], 1)
        out1 = self.mlp(s1*ref)
        out2 = self.mlp(s2*ref)
        out = out1-out2
        #out = self.mlp((s1-s2)*ref)
        #scores = scores.float()
        #print out*scores[0]
        out = out.squeeze()
        return out

class Simple3(torch.nn.Module):
    """
    test
    """
    def __init__(self):
        super(Simple3, self).__init__()
        dim2 = 64
        act_func = 'ReLU'
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(500, dim2))
        self.mlp.add_module('act_func1', nnActi.get_acti(act_func))
#        self.mlp.add_module('bn1', torch.nn.BatchNorm1d(dim2))
        self.mlp.add_module('dp1', torch.nn.Dropout(0.5))
        self.mlp.add_module('fc2', torch.nn.Linear(dim2, 1))


    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        #inp1 = torch.cat([s1, ref], 1)
        #inp2 = torch.cat([s2, ref], 1)
        #out1 = self.mlp(s1*ref)
        #out2 = self.mlp(s2*ref)
        #out = out1-out2
        #out = self.mlp(s1*s2*ref) # 0.05
        #out = self.mlp((s1+s2)*ref) # 0.01/0.17
        #out = self.mlp((s1-s2)*ref) # 0.35/0.45
        #out = self.mlp(2*ref - s1 -s2) # 0.01/0.31
        #out = self.mlp((ref -s1 )* (ref - s2)) # 0.01/0.12
        out = out.squeeze()
        return out

class Simple4(torch.nn.Module):
    """
    conv network
    """
    def __init__(self):
        super(Simple4, self).__init__()
        dim2 = 64
        act_func = 'ReLU'
        self.mlp = torch.nn.Sequential()
        self.conv_layers = torch.nn.Sequential()
        self.conv_layers.add_module('conv1', torch.nn.Conv1d(1, 64, 3, stride = 1))
        self.conv_layers.add_module('dp1', torch.nn.Dropout(0.1))
        self.conv_layers.add_module('af1', nnActi.get_acti('ReLU'))
        self.conv_layers.add_module('conv2', torch.nn.Conv1d(64, 8, 3, stride = 3))
        self.mlp = torch.nn.Sequential()
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(1328, dim2),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(dim2, 1)
                )

    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        #inp1 = torch.cat([s1, ref], 1)
        #inp2 = torch.cat([s2, ref], 1)
        in1 = s1*ref
        in1 = in1.unsqueeze(1)
        in2 = s2*ref
        in2 = in2.unsqueeze(1)
        num_batch = s1.data.shape[0]
        out1 = self.mlp(self.conv_layers(in1).view(num_batch, -1))
        out2 = self.mlp(self.conv_layers(in2).view(num_batch, -1))
        out = out1-out2
        #out = out.squeeze()
        #scores = scores.float()
        #print out*scores[0]
        out = out.squeeze()
        return out

class Simple5(torch.nn.Module):
    """
    more layers input network base
    """
    def __init__(self, num_layers = 2):
        super(Simple5, self).__init__()
        dim2 = 500
        act_func = 'ReLU'
        self.num_layers = num_layers
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(500, dim2))
        self.mlp.add_module('act_func1', nnActi.get_acti(act_func))
#        self.mlp.add_module('bn1', torch.nn.BatchNorm1d(dim2))
        self.mlp.add_module('dp1', torch.nn.Dropout(0.5))
        self.mlp.add_module('fc2', torch.nn.Linear(dim2, 1))
        self.weight_layers = torch.nn.Parameter(torch.FloatTensor(self.num_layers), requires_grad = True)
        self.sf = torch.nn.Softmax()

    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        batch_size, num_layers, num_dim = s1.data.shape
        assert(self.num_layers == num_layers)
        ewl = self.weight_layers.expand(batch_size, self.num_layers) # ==> (batch_size, num)
        ewl = self.sf(ewl).unsqueeze(1) # ==> (batch_size, num_layers)
        #inp1 = torch.cat([s1, ref], 1)
        #inp2 = torch.cat([s2, ref], 1)
        s1 = torch.bmm(ewl, s1).squeeze()
        s2 = torch.bmm(ewl, s2).squeeze()
        ref = torch.bmm(ewl, ref).squeeze()
        out1 = self.mlp(s1*ref)
        out2 = self.mlp(s2*ref)
        out = out1-out2
        #out = self.mlp((s1-s2)*ref)
        #scores = scores.float()
        #print out*scores[0]
        out = out.squeeze()
        return out

class Simple6(torch.nn.Module):
    """
    more layers input distance base 1: with nn
    讨论一下，是什么原因造成的testset的结果比trainingset的差那么多，如果是同一年的数据，是否这种差距会缩小，如果是的话，是否可以通过一部分数据的预训练来提升结果。
    """
    def __init__(self, num_layers = 2):
        """
        当li输出为1维时效果很差，说明一个维度并不足以表达足够的信息。。
        """
        super(Simple6, self).__init__()
        dim2 = 128
        act_func = 'ReLU'
        self.num_layers = num_layers
        self.num_dim = 500
        self.weight_layers = torch.nn.Parameter(torch.ones(self.num_layers), requires_grad = True)
        #self.weight_dimension = torch.nn.Parameter(torch.FloatTensor(self.num_dim), requires_grad = True)
        self.weight_dimension = torch.nn.Parameter(torch.randn(self.num_dim), requires_grad = True)
        self.sf = torch.nn.Softmax()
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(500, dim2))
        self.mlp.add_module('act_func1', nnActi.get_acti(act_func))
#        self.mlp.add_module('bn1', torch.nn.BatchNorm1d(dim2))
        self.mlp.add_module('dp1', torch.nn.Dropout(0.5))
        self.mlp.add_module('fc2', torch.nn.Linear(dim2, 64))
 

    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        # expand the weight parameter
        batch_size, num_layers, num_dim = s1.data.shape
        assert(self.num_layers == num_layers and self.num_dim == num_dim)
        ewl = self.weight_layers.expand(batch_size, self.num_layers) # ==> (batch_size, num_layer)
        ewl = self.sf(ewl).unsqueeze(1) # ==> (batch_size, 1, num_layer)
        #ewd = self.weight_dimension.expand(batch_size, self.num_dim) # ==> (batch_size, num_dim)
        #ewd = self.sf(ewd) # ==> (batch_size, num_dim)
        # process the data with ewl
        s1 = torch.bmm(ewl, s1).squeeze() # ==> (batch_size, num_dim)
        s2 = torch.bmm(ewl, s2).squeeze()
        ref = torch.bmm(ewl, ref).squeeze()
        # process the data with ewd
        #s1 = s1 * ewd
        #s2 = s2 * ewd
        #ref = ref * ewd
        s1 = self.mlp(s1)
        s2 = self.mlp(s2)
        ref = self.mlp(ref)
        # compute the distance 
        d1 = torch.nn.functional.pairwise_distance(s1, ref, p = 1)
        d2 = torch.nn.functional.pairwise_distance(s2, ref, p = 1)
        # d1 = torch.nn.functional.pairwise_distance(s1, ref, p = 2)
        # d2 = torch.nn.functional.pairwise_distance(s2, ref, p = 2)
        #d1 = torch.nn.functional.cosine_similarity(s1, ref)
        #d2 = torch.nn.functional.cosine_similarity(s2, ref)
        out = d2 - d1
        return out

class Simple7(torch.nn.Module):
    """
    more layers input distance base
    """
    def __init__(self, num_layers = 2):
        super(Simple7, self).__init__()
        dim2 = 500
        act_func = 'ReLU'
        self.num_layers = num_layers
        self.num_dim = 500
        #self.weight_layers = torch.nn.Parameter(torch.FloatTensor(self.num_layers), requires_grad = True)
        self.weight_layers = torch.nn.Parameter(torch.randn(self.num_layers), requires_grad = True)
        self.weight_dimension = torch.nn.Parameter(torch.randn(self.num_dim), requires_grad = True)
        self.sf = torch.nn.Softmax(1)

    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        # expand the weight parameter
        batch_size, num_layers, num_dim = s1.data.shape
        assert(self.num_layers == num_layers and self.num_dim == num_dim)
        ewl = self.weight_layers.expand(batch_size, self.num_layers) # ==> (batch_size, num_layer)
        ewl = self.sf(ewl).unsqueeze(1) # ==> (batch_size, 1, num_layer)
        ewd = self.weight_dimension.expand(batch_size, self.num_dim) # ==> (batch_size, num_dim)
        #ewd = self.sf(ewd) # ==> (batch_size, num_dim)
        # process the data with ewl
        s1 = torch.bmm(ewl, s1).squeeze() # ==> (batch_size, num_dim)
        s2 = torch.bmm(ewl, s2).squeeze()
        ref = torch.bmm(ewl, ref).squeeze()
        # process the data with ewd
        #s1 = s1 * ewd
        #s2 = s2 * ewd
        #ref = ref * ewd
        # compute the distance 
        d1 = torch.nn.functional.pairwise_distance(s1, ref, p = 1)
        d2 = torch.nn.functional.pairwise_distance(s2, ref, p = 1)
        #d1 = torch.nn.functional.pairwise_distance(s1, ref, p = 2)
        #d2 = torch.nn.functional.pairwise_distance(s2, ref, p = 2)
        #d1 = torch.nn.functional.cosine_similarity(s1, ref)
        #d2 = torch.nn.functional.cosine_similarity(s2, ref)
        #print '*'*39
        #print self.weight_dimension[:4]
        out = d2 - d1
        return out

class Simple8(torch.nn.Module):
    """
    more layers input distance base 1: with nn
    use capsure in the network
    """
    def __init__(self, num_layers = 2):
        """
        当li输出为1维时效果很差，说明一个维度并不足以表达足够的信息。。
        """
        super(Simple8, self).__init__()
        dim2 = 16
        act_func = 'ReLU'
        self.num_layers = num_layers
        self.num_dim = 500
        self.sf = torch.nn.Softmax()
        self.cap = CapsLayer(self.num_layers, 1, self.num_dim, 64)
        self.mlp = torch.nn.Sequential()
        self.mlp.add_module('fc1', torch.nn.Linear(128, dim2))
        self.mlp.add_module('act_func1', nnActi.get_acti(act_func))
        self.mlp.add_module('dp1', torch.nn.Dropout(0.5))
        self.mlp.add_module('fc2', torch.nn.Linear(dim2, 1))


    def forward(self, s1, s2, ref):
        """
        input1: sent Embeddings
        input2: original target
        """
        # expand the weight parameter
        batch_size, num_layers, num_dim = s1.data.shape
        assert(self.num_layers == num_layers and self.num_dim == num_dim)
        s1 = self.cap(s1)
        s2 = self.cap(s2)
        ref = self.cap(ref)
        d1 = self.mlp(torch.cat([s1, ref], dim = 1))
        d2 = self.mlp(torch.cat([s2, ref], dim = 1))
        # compute the distance 
        d1 = torch.nn.functional.pairwise_distance(s1, ref, p = 1)
        d2 = torch.nn.functional.pairwise_distance(s2, ref, p = 1)
        # d1 = torch.nn.functional.pairwise_distance(s1, ref, p = 2)
        # d2 = torch.nn.functional.pairwise_distance(s2, ref, p = 2)
        #d1 = torch.nn.functional.cosine_similarity(s1, ref)
        #d2 = torch.nn.functional.cosine_similarity(s2, ref)
        out = d2 - d1
        #out = out.squeeze()
        #print out[:5]
        return out

