import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(RandomSampling, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
