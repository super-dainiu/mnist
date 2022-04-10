## Description

All of the dnn module is located in [dnn](./dnn) folder. One may feel free to utilize the layers and optimizers to build his own multi-layer perceptron classifier in the same way as using Pytorch modules.

- [data.py](./dnn/data.py) provides a function to load the MNIST dataset, which is located in [mnist_data](./mnist_data).
- [functional.py](./dnn/functional.py) provides all of the loss function, activation functions and linear layers required for this project. Specially, the Softmax classifier connected to the CrossEntropyLoss does not involve in the backpropagation,
  because the gradient of these two combined
  is easier to compute.
- [nn.py](nn.py) provides a Module template for our net.
- [optim.py](./dnn/optim.py) provides an SGD optimizer.
- [train.py](./dnn/train.py) provides the details of training procedure.
- [utils.py](./dnn/utils.py) provides a DataLoader() class analogous to the torch.utils.data.DataLoader().

The working file [mnist.ipynb](mnist.ipynb) is for hyperparameter searching, and the file [visualizer.ipynb](visualize.ipynb) is for parameter visualization.

A package of python-mnist is supposed to be installed in advance.

```bash
pip install python-mnist
```

## Load Trained Model

Put the downloaded model named *mnist_hidden_256.pkl* (or any other file ends with *.pkl*) under [model](./model) directory.

> 链接：https://pan.baidu.com/s/1ea0penlifIwlBU9KdQZlhw 
> 提取码：0216 

You may refer to the following code to reload the model.

```python
import dnn
import dnn.functional as F
from dnn.data import load_data_mnist
import dnn.nn as nn
import dnn.optim as optim
from dnn.train import train
import numpy as np
import pickle

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, activation='relu'):
        super(MLP, self).__init__()
        if activation == 'relu':
            self.layers = [F.Linear(num_inputs, num_hiddens), F.ReLU(), F.Linear(num_hiddens, num_outputs), F.Softmax()]
        elif activation == 'sigmoid':
            self.layers = [F.Linear(num_inputs, num_hiddens), F.Sigmoid(), F.Linear(num_hiddens, num_outputs), F.Softmax()]
            
with open('./model/mnist_hidden_256.pkl','rb') as f:
    net = pickle.load(f)    # This will be the file
```

