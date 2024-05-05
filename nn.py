class Sequential():
    def __init__(self, layers):
        self.layers = layers

        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x
        
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def step(self, learning_rate):
        for p in self.parameters():
            p.data -= learning_rate * p.grad
