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
    # class Example(nn.Module):
    #     def __init__(self, num_inputs, num_hiddens, num_outputs):
    #         super(Example, self).__init__()
    #         self.layers = [Linear(num_inputs, num_hiddens), ReLU(), Linear(num_hiddens, num_outputs), Softmax()]
    #
    #
    # batch_size, num_input, num_hidden, num_output = 5, 16, 8, 4
    # X = np.random.randint(0, 256, (batch_size, num_input))
    # y = np.random.choice((0, 1, 2, 3), batch_size)
    # net = Example(num_input, num_hidden, num_output)
    # trainer = optim.SGD(net, lr=1e-3)
    # loss = CrossEntropyLoss(net)
    # for i in range(30):
    #     pred_y = net(X)
    #     print(pred_y.argmax(axis=1), y)
    #     L = loss(pred_y, y)
    #     L.backward()
    #     trainer.step()
    #     print(L.item())
    # for layer in L.net.layers:
    #     if layer.requires_grad:
    #         print(f'layer weights: {layer.weight},  layer grads: {layer.weight_grad}')


    class MLP(nn.Module):
        def __init__(self, num_inputs, num_hiddens, num_outputs):
            super(MLP, self).__init__()
            self.layers = [F.Linear(num_inputs, num_hiddens), F.ReLU(), F.Linear(num_hiddens, num_outputs), F.Softmax()]


    train_iter, test_iter = load_data_mnist(64)
    mlp = MLP(784, 512, 10)
    num_epochs, lr = 1000, 1e-2
    loss = F.CrossEntropyLoss(mlp)
    trainer = optim.SGD(mlp, lr=lr)
    history = train(mlp, train_iter, test_iter, loss, trainer, num_epochs)