import numpy as np
import dnn.nn as nn

MAX_GRAD_NORM_W = 1e10
MAX_GRAD_NORM_B = 1e10
MAX_COLOR = 255


class Softmax(object):
    def __init__(self, loss='cross_entropy_error'):
        self.loss = loss
        self.requires_grad = False
        self.output = None
        if self.loss != 'cross_entropy_error':
            raise NotImplementedError

    def forward(self, X: np.ndarray):
        X = X - X.max(axis=-1, keepdims=True)
        self.output = np.divide(np.exp(X), np.exp(X).sum(axis=-1)[:, None])
        return self.output

    def backward(self, grad_output: np.ndarray):
        return grad_output

    def __call__(self, X):
        return self.forward(X)


class ReLU(object):
    def __init__(self):
        self.requires_grad = False
        self.output = None
        self.X = None

    def forward(self, X: np.ndarray):
        self.X = X
        self.output = X * (X > 0)
        return self.output

    def backward(self, grad_output: np.ndarray):
        return grad_output * (self.X > 0)

    def __call__(self, X):
        return self.forward(X)


class Sigmoid(object):
    def __init__(self):
        self.requires_grad = False
        self.output = None

    def forward(self, X: np.ndarray):
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, grad_output: np.ndarray):
        return grad_output * self.output * (1 - self.output)

    def __call__(self, X):
        return self.forward(X)


class Linear(object):
    def __init__(self, num_inputs, num_outputs):
        self.requires_grad = True
        self.weight, self.bias = np.random.randn(num_inputs, num_outputs), np.random.randn(1, num_outputs)
        self.X = None
        self.weight_grad, self.bias_grad = None, None

    def forward(self, X: np.ndarray):
        self.X = X.copy()
        return np.dot(X, self.weight) + self.bias

    def backward(self, grad_output: np.ndarray):
        self.weight_grad = np.dot(self.X.T, grad_output) / self.X.shape[0]
        self.bias_grad = grad_output.mean(axis=0)
        return np.einsum('ij,kj->ik', grad_output, self.weight, optimize=True)

    def __call__(self, X: np.ndarray):
        return self.forward(X)


class CrossEntropyLoss(object):
    def __init__(self, net: nn.Module, classifier='softmax'):
        self.requires_grad = False
        self.batch_size = None
        self.cls = None
        self.label = None
        self.classifier = classifier
        self.output = None
        self.net = net
        if self.classifier != 'softmax':
            raise NotImplementedError

    def forward(self, pred: np.ndarray, label: np.ndarray):
        self.batch_size, self.cls = pred.shape
        self.label = label
        if self.classifier == 'softmax':
            self.output = pred
        return self

    def backward(self):
        if self.classifier == 'softmax':
            grad_output = self.output.copy()
            grad_output[np.arange(self.batch_size), self.label] -= 1
            self.net.backward(grad_output / self.cls)

    def item(self):
        return - np.log(self.output[np.arange(self.batch_size), self.label] + 1e-7).mean()

    def __call__(self, pred: np.ndarray, label: np.ndarray):
        return self.forward(pred, label)

