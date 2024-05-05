import torch
import torch.nn.functional as F

# objective function to minimize, defined as cross-entropy loss.
def objective(scores, ys):
    return F.cross_entropy(scores, ys)

def get_loss(model, x, y, training):
    scores = model.forward(x, training)
    loss = objective(scores, y)
    return loss

def optimize(model, xs, ys, nepoch=100, nbatch=30, learning_rate=0.1):
    losses = {
        'train': [],
        'valid': [],
    }
    for i in range(nepoch):
        batch_indices = torch.randint(0, xs['train'].shape[0], (nbatch,))
        x, y = xs['train'][batch_indices], ys['train'][batch_indices]
        tloss = get_loss(model, x, y, training=True)
        losses['train'].append(tloss.item())

        with torch.no_grad():
            batch_indices = torch.randint(0, xs['valid'].shape[0], (nbatch,))
            x, y = xs['valid'][batch_indices], ys['valid'][batch_indices]
            vloss = get_loss(model, x, y, training=False)
            losses['valid'].append(vloss.item())

        model.zero_grad()
        tloss.backward()
        model.step(learning_rate)
        
    return losses
