class Module(object):
    def __init__(self):
        self.layers = []

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, grad_output=None):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def __call__(self, X):
        return self.forward(X)
