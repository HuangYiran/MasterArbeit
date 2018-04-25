import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsLayer(nn.Module):
    def __init__(self, in_features, out_features, in_dim, out_dim, T = 3):
        """
        input: 
            in_features: number of the input features
            out_features: number of the output features
            in_dim: number of dimension of the input vector for each capsure
            out_dim: number of dimension of the output vector for each capsure
            T: times of dynary routing 
        """
        super(Capslayer, self).__init__()
        self.in_fs = in_features
        self.out_fs = out_features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.T = T
        self.W = nn.Parameter(1, self.in_fs, self.out_fs, self.in_dim, self.out_dim) # share W in the batch

    def forward(self, x):
        # x.shape:(batch_size, in_fs, in_dim)
        batch_size, in_fs, in_dim = x.shape
        assert(self.in_fs == in_fs& self.in_dim == in_dim)
        # change x to (batch_size, in_fs, out_fs, 1, in_dim)
        x = torch.cat([x]*self.out_fs, dim = 2).unsqueeze(dim = 3)
        # expand the w to the hold batch
        W = torch.cat([self.W]*batch_size, dim = 0)
        # u = x*w with the shape (batch_size, in_fs, out_fs, 1, out_dim)
        u_hat = torch.matmul(x, W)
        # set b with shape (1, in_fs, out_fs, 1) as W shared in batch
        b = torch.zeros([1, self.in_fs, self.out_fs, 1]).double()
        b = torch.autograd.Variable(b)
        for i in range(self.T):
            # c shape (1, in_fs, out_fs, 1)
            c = F.softmax(b, dim = 2) # along the out_features dimension, because first dim is 1
            # expand to the hold batch, and add a dimension
            c = torch.cat([c]*batch_size, dim = 0).unsqueeze(dim = 4)
            # s = u*c ,and sum the in_fs dimension. shape (batch_size, 1, out_fs, 1, out_dim)
            s = (u_hat * c).sum(dim = 1, keepdim =True)
            # do the squash, output with the same shape as the input (batch_size, 1, out_fs, 1, out_dim)
            v = squash(s, dim = -1)
            # expand the v for each input features
            v_1 = torch.cat([v]*self.in_fs, dim = 1)
            # recompute the b 
            up_b = torch.matmul(u_hat, v_1.transpose(3,4))
            b = b + up_b
        return v.squeeze()


##################
# add function
##################
def squash(x, dim = -1):
    """
    complish the function squash y = ||x||^2/(1+||x||^2)*x/||x||
    """
    x_squ = torch.sum(x**2, dim = dim, keepdim = True)
    x_sqrt = torch.sqrt(x_squ)
    return x_squ/(1.0 + x_sqrt)*x/x_sqrt
