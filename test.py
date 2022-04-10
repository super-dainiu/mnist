import dnn
import dnn.functional as F
from dnn.data import load_data_mnist
import dnn.nn as nn
import dnn.optim as optim
from dnn.train import train
import time
import numpy as np
from dnn.functional import *

if __name__ == "__main__":
    class Example(nn.Module):
        def __init__(self, num_inputs, num_hiddens, num_outputs):
            super(Example, self).__init__()
            self.layers = [Linear(num_inputs, num_hiddens), ReLU(), Linear(num_hiddens, num_outputs), Softmax()]


    batch_size, num_input, num_hidden, num_output = 64, 1024, 512, 4
    X = np.random.randint(0, 256, (batch_size, num_input))
    y = np.random.choice((0, 1, 2, 3), batch_size)
    net = Example(num_input, num_hidden, num_output)
    loss = CrossEntropyLoss(net)
    pred_y = net(X)
    print(pred_y.argmax(axis=1), y)
    L = loss(pred_y, y)
    L.backward()
    print(L.item())
    for layer in L.net.layers:
        if layer.requires_grad:
            print(f'layer weights: {layer.weight},  layer grads: {layer.weight_grad}')