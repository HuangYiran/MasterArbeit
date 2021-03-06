#-*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as fn

#class CorrLoss(torch.autograd.Function):
class MarginLoss(nn.Module):
    """
    L_c = T_c max(0, m+ -||v||)^2 + \lambda (1-T_c)max(0, ||v|| - m-)^2 
    """
    def __init__(self, m_plus=0.9, m_minus=0.1, lamb = 0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lamb = lamb

    def forward(self, o, t):
        """
        input:
            o: should be output of capsure with the shape (batch_size, num_class, num_dim)
            t: target class with the shape (batch_size,)
        """
        # set attribute 
        batch_size, num_class, num_dim = o.shape
        zero = torch.autograd.Variable(torch.zeros([1]).double())
        lamb = torch.autograd.Variable(torch.Tensor([self.lamb]).double())
        # transform the int target value to one-hot. if the shape of o is wrong then the program callapse
        t_hot = torch.zeros([batch_size, num_class])
        for index, value in t:
            t_hot[index, value] = 1
        # now t_hot ''s shape is (batch_size, num_class)
        # calcute the norm of o: shape (batch_size, num_class)
        o_norm = torch.sqrt(torch.sum(o**2, dim = 2, keepdim = True)).squeeze()
        # calcute the loss
        L = torch.max(zero, self.m_plus - o_norm)**2
        R = torch.max(zero, o_norm - self.minus)**2
        loss = torch.sum(t*L + lamb*(1-t)*R, dim = 1)
        return loss.mean()

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

class Product(nn.Module):
    def forward(self, o, t):
        return torch.mean(-1*o*t)

class MSECorrLoss(nn.Module):
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
        corr = self.corrLoss(o, t)
        loss = mse + self.p * corr
        return loss

class PReLULoss(nn.Module):
    """
    use PReLU as the loss function
    wish that when the o smaller as t, it get more (or less) punishment 
    """
    def __init__(self, p = 2):
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
class PReLUCorrLoss(nn.Module):
    """
    PReLU + corr
    """
    def __init__(self, p_rate = 2, gate_rate = 1.5):
        super(PReLUCorrLoss, self).__init__()
        self.gate_rate = gate_rate
        self.ploss = PReLULoss(p_rate)
        self.corr = CorrLoss()
    
    def forward(self, o, t):
        prelu = self.ploss(o, t)
        corr = self.corr(o, t)
        loss = prelu + self.gate_rate * corr
        return loss

class VAELoss(nn.Module):
    """
    mse + KL
    """
    def forward(self, re_x, x, mu, logvar):
        mse = torch.nn.MSELoss(size_average = True)
        kld = torch.nn.KLDivLoss(size_average = True)
        BCE = mse(re_x, x)
        KLD = kld(re_x, x)
        #KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(logvar)#????????
        #KLD = torch.sum(KLD_element).mul_(-0.5)
        return BCE + KLD
