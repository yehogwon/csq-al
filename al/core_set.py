import numpy as np
import pdb
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
from sklearn.metrics import pairwise_distances

from torch.utils.data import Subset

def furthest_first(X, X_set, n):
    m = np.shape(X)[0]
    if np.shape(X_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(X, X_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for i in range(n):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(X, X[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return idxs

class CoreSet(Strategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args, tor=1e-4):
        super(CoreSet, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def query(self, n):
        t_start = datetime.now()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        lb_flag = self.idxs_lb.copy()
        embedding = self.predict(self.train_raw_dataset, return_prob=False, return_embedding=True)
        embedding = embedding.numpy()

        chosen = furthest_first(embedding[idxs_unlabeled, :], embedding[lb_flag, :], n)

        return idxs_unlabeled[chosen]
