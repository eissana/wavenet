import torch
import numpy as np
import const

def read_names(filename):
    with open(filename) as f:
        names = f.read().splitlines()
    return names

# return input embedding (N x nblock) and output labels of size N
def load(names, nblock):
    xs, ys = [], []
    for name in names:
        # padding with zero (which represents ^)
        x = [0]*nblock
        for ch in name + const.end:
            i = const.atoi[ch]
            xs.append(x)
            ys.append(i)
            # pop from front and push from back (fixed-size queue).
            x = x[1:] + [i]
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys

def split(x, y, train_ratio, valid_ratio):
    ntrain = int(train_ratio*x.shape[0])
    nvalid = int(valid_ratio*x.shape[0])
    perm = np.random.permutation(x.shape[0])

    splitx = {
        'train': x[perm][:ntrain],
        'valid': x[perm][ntrain:ntrain+nvalid],
        'test': x[perm][ntrain+nvalid:],
    }
    splity = {
        'train': y[perm][:ntrain],
        'valid': y[perm][ntrain:ntrain+nvalid],
        'test': y[perm][ntrain+nvalid:],
    }
    return splitx, splity
