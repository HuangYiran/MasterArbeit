import sys
sys.path.append('./utils')
sys.path.append('./models')
import torch
import argparse
import numpy as np

from Autoencoder import VAE, Autoencoder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-model', default = 'autoencoder')
parser.add_argument('-checkpoint', default = './checkpoints/autoencoder')

datasets = ['/tmp/_sumde.en2015decoder_hidden.ref.npy', '/tmp/_sumde.en2015decoder_hidden.s1.npy','/tmp/_sumde.en2015decoder_hidden.s2.npy',
        '/tmp/_sumde.en2016decoder_hidden.ref.npy', '/tmp/_sumde.en2016decoder_hidden.s1.npy', '/tmp/_sumde.en2016decoder_hidden.s2.npy']

def main():
    opt = parser.parse_args()
    for dataset in tqdm(datasets):
        ds = torch.from_numpy(np.load(dataset))
        ds = torch.autograd.Variable(ds, requires_grad = False)
        model = torch.load(opt.checkpoint)
        if opt.model == 'vae':
            out = get_hidden_VAE(model, ds)
        elif opt.model == 'autoencoder':
            out = get_hidden_Autoencoder(model, ds)
        else:
            print('unrecognized model type')
        # write data
        np.save(dataset+'_af_'+opt.model+'_auto', out.data.numpy())

def get_hidden_VAE(model, data):
    mu, logvar = model.encode(data)
    out = model.reparametrize(mu, logvar)
    return out

def get_hidden_Autoencoder(model, data):
    out = model.encoder(data)
    return out


if __name__ == '__main__':
    main()
