import torch

class Embedding():
    def __init__(self, num_embeddings, embedding_dim):
        self.emb = torch.randn(num_embeddings, embedding_dim)

    def forward(self, x, training):
        self.out = self.emb[x]
        return self.out

    def parameters(self):
        return [self.emb]

class Flatten():
    def __init__(self, last_dim_multiplier):
        self.last_dim_multiplier = last_dim_multiplier

    def forward(self, x, training):
        if x.ndim == 3:
            self.out = x.view(x.shape[0], -1, self.last_dim_multiplier*x.shape[2])
            if self.out.shape[1] == 1:
                self.out = self.out.squeeze(1)
        elif x.ndim == 2:
            self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []

class Linear():
    def __init__(self, fan_in, fan_out, bias=False, gain=1.0):
        self.W = torch.randn(fan_in, fan_out) * gain * fan_in**-0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def forward(self, x, training):
        # storing it for analysis purposes.
        self.out = x @ self.W
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.W] + ([] if self.bias is None else [self.bias])

class BatchNorm1d():
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        self.scale = torch.ones(1, num_features)
        self.shift = torch.zeros(1, num_features)

        self.eps = eps
        self.momentun = momentum

        # Running mean and variance to be used in inference.
        # These are not part of the computation graph, so no need to compute gradient for these.
        self.running_mean = torch.zeros(1, num_features)
        self.running_var = torch.ones(1, num_features)

    def forward(self, x, training):
        if training:
            # reduce on all dimensions except the last one.
            dims = tuple(range(x.ndim-1))
            mean = x.mean(axis=dims, keepdims=True)
            var = x.var(axis=dims, keepdims=True)
        else:
            mean = self.running_mean
            var = self.running_var

        if training:
            with torch.no_grad():
                self.running_mean = (1-self.momentun) * self.running_mean + self.momentun * mean
                self.running_var = (1-self.momentun) * self.running_var + self.momentun * var

        # Standardize and perturb by scaling and shifting.
        # storing it for analysis purposes.
        self.out = self.scale * (x - mean) * (self.running_var + self.eps)**-0.5 + self.shift
        return self.out

    def parameters(self):
        return [self.scale, self.shift]

class Tanh():
    def forward(self, x, training):
        # storing it for analysis purposes.
        self.out = x.tanh()
        return self.out

    def parameters(self):
        return []
