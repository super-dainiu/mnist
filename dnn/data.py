from mnist import MNIST
from dnn.utils import DataLoader
import time


def load_data_mnist(batch_size=64, shuffle=True):
    mndata = MNIST('./mnist_data')
    train_iter = DataLoader(mndata.load_training(), batch_size, shuffle)
    test_iter = DataLoader(mndata.load_testing(), batch_size)
    print(f'Loading MNIST dataset. {len(train_iter)} items for training, {len(test_iter)} items for testing. batch_size={batch_size}')
    print(f'X size: {train_iter.images.shape[1]}, MNIST image scale 28x28, total_cls=10.')
    return train_iter, test_iter, train_iter.images.shape[1], 10


if __name__ == "__main__":
    train, test, _, _ = load_data_mnist()
    for iterator, name in ((train, 'train_iter'), (test, 'test_iter')):
        t = time.time()
        for _ in iterator:
            continue
        print(f'{time.time()-t} for {name}')