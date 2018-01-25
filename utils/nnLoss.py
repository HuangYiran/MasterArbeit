#-*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as fn

#class CorrLoss(torch.autograd.Function):
class CorrLoss(nn.Module):
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
        #print('o.size {}, t.size {}'.format(o.size(), t.size()))
        assert(o.size() == t.size())
        # calcu z-score for o and t
        o_m = o.mean(dim = 0)
        o_s = o.std(dim = 0)
        o_z = (o - o_m)/o_s
        #print('o_z:',o_z.size())

        t_m = t.mean(dim =0)
        t_s = t.std(dim = 0)
        t_z = (t - t_m)/t_s
        #print('t_z:', t_z.size())
        # calcu corr between o and t
        tmp = o_z * t_z
        #print('tmp:', tmp.size())
        corr = tmp.mean(dim = 0)
        #print corr.size()
        return  1 - corr

class MESCorrLoss(nn.Module):
    """
    use MSE - p * Corr as the loss, p is the 
    a problem is the value of MSE dependend on the target value and the value of Corr is [-1, 1]. so i don't know if it is sinnful to do the coordination
    """
    def __init__(self, p = 1.5):
        super(MSECorrLoss, self).__init__()
        self.p = p
        self.mseLoss = nn.MSELoss()
        self.corrLoss = CorrLoss()
        
    def forward(self, o, t):
        mse = self.mseLoss(o, t)
        corr = 1- self.corrLoss(o, t)
        loss = mse - corr
        return loss

class PReLULoss(nn.Module):
    """
    use PReLU as the loss function
    wish that when the o smaller as t, it get more (or less) punishment 
    """
    def __init__(self, p = 12):
        super(PReLULoss, self).__init__()
        self.prelu = nn.PReLU(1, p)
        for param in self.prelu.parameters():
            param.requires_grad = False
    
    def forward(self, o, t):
        loss = self.prelu(o - t)
        return torch.mean(torch.abs(loss))
        """
        dis = o - t
        batch_size = dis.size()[0]
        zero = torch.rand([1]).fill_(0)
        loss = 0
        # print dis # 这里涉及到首个Evaluation 的问题，所以应该另外考虑 可以单纯跳了先
        #print '=====' 
        for i in range(batch_size):
            nan = torch.sqrt(torch.FloatTensor([-1])) # don't know how to create a nan Tensor directly
            # can not use torch.equal to compare the nan value
            print(dis[i], nan)
            if torch.equal(dis[i].data, nan):
                print 'equal'
                loss = loss + 10
                continue
            print 'unequal'
            combined = torch.cat((dis[i], zero), 0)
            #print combined
            #print '-----------;'
            loss += torch.max(combined) - self.p * torch.min(combined)
        return loss/batch_size
        """