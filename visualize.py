import torch
import matplotlib.pyplot as plt
import layer

def plot(model, layer_type):
    plt.figure(figsize=(20, 5))
    legends = []
    for i, l in enumerate(model.layers[:-1]):
        if not isinstance(l, layer_type):
            continue
        data = l.out.detach()
        print(f'layer: {i:3d}, mean: {data.mean():+.3f}, std: {data.std():.3f}', end='')
        if layer_type == layer.Tanh:
            print(f', saturation: {(data.abs() > 0.97).float().mean()*100:2.2f}%')
        else:
            print()
        hy, hx = torch.histogram(data)
        plt.plot(hx[:-1], hy);
        legends.append(f'Layer {i}')
    plt.legend(legends, loc='best');


def plot_ratios(model, ratios):
    plt.figure(figsize=(20,10))
    nratios = len(ratios)
    legends = []
    for i, p in enumerate(model.parameters()):
        plt.plot([ratios[j][i] for j in range(nratios)])
        legends.append(f'Layer {i}')
    # line corresponds to 0.001 where the curve should ideally converge too.
    plt.plot([-0.02*nratios, 1.02*nratios], [-3, -3], 'k')
    plt.legend(legends, loc='best');
