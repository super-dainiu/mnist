import numpy as np


class DataLoader(object):
    def __init__(self, data, batch_size=64, shuffle=True):
        self.images, self.labels = np.array(data[0]).reshape(len(data[0]), -1)/255, np.array(data[1]).reshape(len(data[1]), -1)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            return self.images[item], self.labels[item]
        if isinstance(item, slice):
            return self.images[item], self.labels[item]
        if isinstance(item, int):
            if item < 0:
                item += len(self)
            if item < 0 or item >= len(self):
                raise (IndexError, "The index (%d) is out of range." % item)
            return self.images[item], self.labels[item]

    def __iter__(self):
        idx = np.arange(len(self))
        np.random.shuffle(idx)
        self.images, self.labels = self[idx]
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            batch = self.images[self.n:min(len(self), self.n+self.batch_size)], self.labels[self.n:min(len(self), self.n+self.batch_size)]
            self.n += self.batch_size
            return batch
        else:
            raise StopIteration


