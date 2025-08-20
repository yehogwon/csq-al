from .strategy import Strategy
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
import pdb
from scipy import stats
import numpy as np

from torch.utils.data import Subset

def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist

# k-means++ initialization
def init_centers(X1, X2, chosen, chosen_list,  mu, D2, device='cpu', deterministic=False):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0

        Ddist = (D2 ** 2) / np.sum(D2 ** 2)
        Ddist = np.nan_to_num(Ddist, nan=0.0, posinf=0.0, neginf=0.0)
        Ddist = Ddist / np.sum(Ddist)

        # Debugging and validation
        if not np.isclose(np.sum(Ddist), 1.0):
            raise ValueError(f'The sum of provided pk is not 1: {np.sum(Ddist)} | {np.isnan(Ddist).any()} | {np.isinf(Ddist).any()} | {np.min(Ddist)} | {np.max(Ddist)}')

        if deterministic:
            sorted_dist_indices = np.argsort(Ddist)
            added = False
            for i in sorted_dist_indices:
                if i not in chosen:
                    ind = i
                    added = True
                    break
            if not added: 
                raise ValueError('No sample to add')
        else:
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in chosen: ind = customDist.rvs(size=1)[0]
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
    return chosen, chosen_list, mu, D2

class BadgeSampling(Strategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(BadgeSampling, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.deterministic = args['deterministic']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        if len(idxs_unlabeled) <= n:
            return idxs_unlabeled
        probs, embs = self.predict(Subset(self.train_raw_dataset, idxs_unlabeled), return_prob=True, return_embedding=True)
        probs = probs.numpy()
        embs = embs.numpy()

        # the logic below reflects a speedup proposed by Zhang et al.
        # see Appendix D of https://arxiv.org/abs/2306.09910 for more details
        m = (~self.idxs_lb).sum()
        mu = None
        D2 = None
        chosen = set()
        chosen_list = []
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        probs = -1 * probs
        probs[np.arange(m), max_inds] += 1
        prob_norms_square = np.sum(probs ** 2, axis=-1)
        for _ in range(n):
            chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2, device=self.device, deterministic=self.deterministic)
        return idxs_unlabeled[chosen_list]
