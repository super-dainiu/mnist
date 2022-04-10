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
        # X_copy = X.copy()
        X = X - X.max(axis=-1, keepdims=True)
        # print(X, X_copy)
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
        self.X = X.copy()
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
        # self.weight, self.bias = self.weight / np.linalg.norm(self.weight), self.bias / np.linalg.norm(self.bias)
        self.X = None
        self.weight_grad, self.bias_grad = None, None

    def forward(self, X: np.ndarray):
        # print(f'layer X norm : {np.linalg.norm(X)}')
        self.X = X.copy()
        # print(f'{np.dot(X, self.weight)}, {self.bias}')
        return np.dot(X, self.weight) + self.bias

    def backward(self, grad_output: np.ndarray):
        self.weight_grad = np.dot(self.X.T, grad_output) / self.X.shape[0]
        self.bias_grad = grad_output.mean(axis=0)
        norm_weight = np.linalg.norm(self.weight_grad)
        # if norm_weight > MAX_GRAD_NORM_W:
        #     self.weight_grad = self.weight_grad / norm_weight * MAX_GRAD_NORM_W
        # norm_bias = np.linalg.norm(self.bias_grad)
        # if norm_bias > MAX_GRAD_NORM_B:
        #     self.bias_grad = self.bias_grad / norm_bias * MAX_GRAD_NORM_B
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
            # print(grad_output)
            grad_output[np.arange(self.batch_size), self.label] -= 1
            # print(grad_output)
            self.net.backward(grad_output / self.cls)

    def item(self):
        return - np.log(self.output[np.arange(self.batch_size), self.label] + 1e-7).mean()

    def __call__(self, pred: np.ndarray, label: np.ndarray):
        return self.forward(pred, label)


# if __name__ == "__main__":
#     class Example(nn.Module):
#         def __init__(self, num_inputs, num_hiddens, num_outputs):
#             super(Example, self).__init__()
#             self.layers = [Linear(num_inputs, num_hiddens), ReLU(), Linear(num_hiddens, num_outputs), Softmax()]
#
#     batch_size, num_input, num_hidden, num_output = 64, 1024, 512, 4
#     X = np.random.randint(0, 256, (batch_size, num_input))
#     y = np.random.choice((0, 1, 2, 3), batch_size)
#     net = Example(num_input, num_hidden, num_output)
#     loss = CrossEntropyLoss(net)
#     pred_y = net(X)
#     print(pred_y.argmax(axis=1), y)
#     L = loss(pred_y, y)
#     L.backward()
#     print(L.item())
#     for layer in L.net.layers:
#         if layer.requires_grad:
#             print(f'layer weights: {layer.weight},  layer grads: {layer.weight_grad}')
