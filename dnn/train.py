import dnn.nn as nn
import dnn.functional as functional
import dnn.optim as optim
from dnn.utils import DataLoader
from tqdm import tqdm
import numpy as np


def train(net: nn.Module, train_iter: DataLoader, test_iter: DataLoader, loss: functional.CrossEntropyLoss,
          trainer: optim.SGD, num_epochs, inter_show=5):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for i in tqdm(range(num_epochs)):
        total_acc = []
        total_loss = []
        for j, (X, y) in enumerate(train_iter):
            y = np.array(y).flatten()
            logits = net(X)
            L = loss(logits, y)
            y_pred = logits.argmax(axis=1)
            total_loss.append(L.item())
            total_acc.append((y_pred == y).mean())
            L.backward()
            trainer.step()
        train_loss.append(sum(total_loss) / len(total_loss))
        train_acc.append(sum(total_acc) / len(total_acc))

        total_acc = []
        total_loss = []
        for X, y in test_iter:
            y = np.array(y).flatten()
            logits = net(X)
            L = loss(logits, y)
            y_pred = logits.argmax(axis=1)
            total_loss.append(L.item())
            total_acc.append((y_pred == y).mean())
        test_loss.append(sum(total_loss) / len(total_loss))
        test_acc.append(sum(total_acc) / len(total_acc))

        if (i + 1) % inter_show == 0:
            print(
                f"result of epoch {i + 1}, train loss:{train_loss[i]}, train accuracy:{train_acc[i] * 100}%\n"
                f"test loss:{test_loss[i]}, test accuracy:{test_acc[i] * 100}%")
    return train_loss, train_acc, test_loss, test_acc
