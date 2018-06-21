import sys
sys.path.append('utils/')
sys.path.append('models/')
import torch
import argparse
import numpy as np

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from Autoencoder import *
from nnLoss import *

parser = argparse.ArgumentParser()
parser.add_argument('-data', default = '/tmp/2015_sum_ref')
parser.add_argument('-model', default = 'vae')
parser.add_argument('-batch_size', default = 100)
parser.add_argument('-save_dir', default = './checkpoints')


def main():
    opt = parser.parse_args()
    # set models and loss
    if opt.model == 'vae':
        model = VAE()
        loss = VAELoss()
    elif opt.model == 'autoencoder':
        model = Autoencoder()
        loss = torch.nn.MSELoss()
    else:
        print('unrecognized model type')
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # set lr scheduler
    lamb1 = lambda x: .1**(x//30)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lamb1)
    # read data
    data = Data(opt.data)
    dataloader = DataLoader(data, batch_size = opt.batch_size, shuffle = True)
    # train the model 
    num_epochs = 100
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        for batch_idx, dat in enumerate(dataloader):
            x = torch.autograd.Variable(dat, requires_grad = False)
            optimizer.zero_grad()
            if opt.model == 'vae':
                re_x, mu, logvar = model(x)
                lo = loss(re_x, x, mu, logvar)
            elif opt.model == 'autoencoder':
                re_x = model(x)
                lo = loss(re_x, x)
            lo.backward()
            train_loss += lo.data[0]
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(x),
                    len(dataloader.dataset), 100.*batch_idx/len(dataloader),
                    lo.data[0]/len(x)
                    ))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch,
                train_loss/len(dataloader.dataset)
                ))
    torch.save(model.state_dict(), opt.save_dir+'/'+opt.model)

class Data(torch.utils.data.Dataset):
    def __init__(self, dir):
        super(Data, self).__init__()
        self.data = torch.from_numpy(np.load(dir))
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

if __name__ == '__main__':
    main()
