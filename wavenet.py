import torch
import numpy as np
import matplotlib.pyplot as plt

import const
import layer
import nn
import util
import data
import metrics

if __name__ == '__main__':
    nblock = 8

    names = data.read_names('data/names.txt')
    x, y = data.load(names, nblock)
    xs, ys = data.split(x, y, train_ratio=0.8, valid_ratio=0.1)
    
    nembd = 24
    nhidden = 128

    tanh_gain = 5/3 
    out_gain = 0.1

    layers = [
        # in: [N, nblock] out: [N, nblock, nembd]
        layer.Embedding(const.nchars, nembd),
        # in: [N, nblock, nembd] out: [N, nblock//2, 2*nembd]
        layer.Flatten(2),
        # in: [N, nblock//2, 2*nembd] out: [N, nblock//2, nhidden]
        layer.Linear(2*nembd, nhidden, gain=tanh_gain), 
        # in: [N, nblock//2, 2*nembd] out: [N, nblock//2, nhidden]
        layer.BatchNorm1d(nhidden), 
        # in: [N, nblock//2, 2*nembd] out: [N, nblock//2, nhidden]
        layer.Tanh(),
        # in:  [N, nblock//2, nhidden] out: [N, nblock//4, 2*nhidden]
        layer.Flatten(2),
        # in: [N, nblock//4, 2*nhidden] out: [N, nblock//4, nhidden]
        layer.Linear(2*nhidden, nhidden, gain=tanh_gain), 
        # in: [N, nblock//4, 2*nhidden] out: [N, nblock//4, nhidden]
        layer.BatchNorm1d(nhidden), 
        # in: [N, nblock//4, 2*nhidden] out: [N, nblock//4, nhidden]
        layer.Tanh(),
        # in: [N, nblock//4, nhidden] out: [N, 2*nhidden] (nblock//8==1)
        layer.Flatten(2),
        # in: [N, 2*nhidden] out: [N, nhidden]
        layer.Linear(2*nhidden, nhidden, gain=tanh_gain), 
        # in: [N, nhidden] out: [N, nhidden]
        layer.BatchNorm1d(nhidden), 
        # in: [N, nhidden] out: [N, nhidden]
        layer.Tanh(),
        # in: [N, nhidden] out: [N, nchars]
        layer.Linear(nhidden, const.nchars, gain=out_gain),
    ]
    model = nn.Sequential(layers)
    print(f'model parameters: {util.nparameters(model)}')
    print(f'model layers:\n {util.layer_sizes(model, xs['train'])}')
    
    nepoch = 1000
    losses = metrics.optimize(model, xs, ys, nepoch=nepoch, nbatch=32, learning_rate=0.1)
    
    last_1p = int(0.1 * nepoch)
    print(f'final training loss: {np.mean(losses['train'][-last_1p:])}')
    print(f'final validation loss: {np.mean(losses['valid'][-last_1p:])}')
    
    # plt.plot(tlosses);
    # plt.plot(vlosses);
    plt.plot(torch.tensor(losses['train']).view(-1, last_1p).mean(axis=1));
    plt.plot(torch.tensor(losses['valid']).view(-1, last_1p).mean(axis=1));
    plt.legend(['training loss', 'validation loss'])
    plt.show()
