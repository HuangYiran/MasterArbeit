import torch
import torch.nn as nn
import torch.nn.functional as fn

class CorrLoss(torch.autograd.Function):
    """
    use 1 - correlational coefficience between the output of the network and the target as the loss
    input (o, t):
        o: Variable of size (batch_size, 1) output of the network
        t: Variable of size (batch_size, 1) target value
    output (corr):
        corr: Variable of size (1)
    """
    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, o, t):
        print('o.size {}, t.size {}'.format(o.size(), t.size()))
        assert(o.size() == t.size())
        # calcu z-score for o and t
        o_m = o.mean(dim = 0)
        o_s = o.std(dim = 0)
        o_z = (o - o_m)/o_s
        print('o_z:',o_z.size())

        t_m = t.mean(dim =0)
        t_s = t.std(dim = 0)
        t_z = (t - t_m)/t_s
        print('t_z:', t_z.size())
        # calcu corr between o and t
        tmp = o_z * t_z
        print('tmp:', tmp.size())
        corr = tmp.mean(dim = 0)
        print corr.size()
        return  1 - corr
