#-*- coding: UTF-8 -*-

########### 准备数据
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import  Variable
import torch.nn as nn
x = np.random.randn(1000, 1)*4
w = np.array([0.5,])
bias = -1.68

y_true = np.dot(x, w) + bias  #真实数据
y = y_true + np.random.randn(x.shape[0])#加噪声的数据
#我们需要使用x和y，以及y_true回归出w和bias

############# 定义回归网络的类
class LinearRression(nn.Module):
    def __init__(self, input_size, out_size):
        super(LinearRression, self).__init__()
        self.x2o = nn.Linear(input_size, out_size)
        self.db = torch.autograd.Variable(torch.ones(input_size)*2) # double the input 为了测试！！！！！！！！
    #初始化
    def forward(self, x):
        x = self.db * x
        return self.x2o(x)
    #前向传递

cuda = False
############## 接下来介绍将定义模型和优化器
batch_size = 10
model = LinearRression(1, 1)#回归模型
criterion = nn.MSELoss()  #损失函数
#调用cuda
if cuda:
    model.cuda()
    criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
losses = []


# 下面就是训（练）练（丹）了 
epoches = 100
for i in range(epoches):
    loss = 0
    optimizer.zero_grad()#清空上一步的梯度
    idx = np.random.randint(x.shape[0], size=batch_size)
    batch_cpu = Variable(torch.from_numpy(x[idx])).float()
    if cuda:
        batch = batch_cpu.cuda()#很重要
    else:
        batch = batch_cpu

    target_cpu = Variable(torch.from_numpy(y[idx])).float()
    if cuda:
        target = target_cpu.cuda()#很重要
    else:
        target = target_cpu
    output = model.forward(batch)
    loss += criterion(output, target)
    loss.backward()
    optimizer.step()

    if (i +1)%10 == 0:
        print('Loss at epoch[%s]: %.3f' % (i, loss.data[0]))
    losses.append(loss.data[0])

plt.plot(losses, '-or')
plt.xlabel("Epoch")
plt.xlabel("Loss")

plt.show()