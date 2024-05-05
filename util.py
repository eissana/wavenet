import torch
import torch.nn.functional as F
import pandas as pd

import const

def nparameters(model):
    '''
    Returns the total number of model parameters.
    '''
    return sum([p.nelement() for p in model.parameters()])

def layer_sizes(model, x):
    '''
    Returns each layer's input and output sizes as a dataframe.
    '''
    names = []
    ins, outs = [], []
    for layer in model.layers:
        ins.append(list(x.shape))
        names.append(layer.__class__.__name__)
        x = layer.forward(x, training=False)
        outs.append(list(x.shape))
    result = pd.DataFrame()
    result['layer-name'] = names
    result['input-size'] = ins
    result['output-size']= outs
    return result

def ratio(model, learning_rate):
    '''
    Returns the ration of gradient directions to the points in log scale.
    '''
    out = []
    with torch.no_grad():
        for p in model.parameters():
            grad_dir = learning_rate * p.grad.std()
            out.append((grad_dir / p.data.std()).log10().item())
    return out

def gen_word(model, nblock):
    '''
    Generates new words using the trained model.
    '''
    x = [0]*nblock
    out = ''
    while True:
        score = model.forward(torch.tensor([x]), training=False)
        score = F.softmax(score, dim=1)
        j = score.multinomial(1)
        ch = const.itoa[j]
        if ch == const.end:
            break
        x = x[1:] + [j.item()]
        out += ch
    return out
