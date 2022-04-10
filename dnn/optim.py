import numpy as np
from tqdm import tqdm
import dnn.nn as nn
import dnn.functional as functional


class SGD(object):
    def __init__(self, net: nn.Module, lr=0.001, decay=None):
        self.lr = lr
        self.decay = decay
        self.net = net

    def step(self):
        for layer in self.net.layers:
            if layer.requires_grad:
                grad = layer.weight_grad.copy()
                # print(np.linalg.norm(grad), np.linalg.norm(layer.weight))
                if self.decay is not None:
                    grad += layer.weight * self.decay
                layer.weight -= self.lr * grad

                grad = layer.bias_grad.copy().reshape(1, -1)
                if self.decay is not None:
                    grad += layer.bias * self.decay
                layer.bias -= self.lr * grad