from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from .strategy import Strategy
import wandb
from tqdm import tqdm

from torch.utils.data import Subset

import math

from .strategy_dtopk import DynamicTopKStrategy, clip_k
from .badge_sampling import init_centers, distance

from scipy import stats
from sklearn.metrics import pairwise_distances
import pandas as pd

import os

from time import time

# Construct the candidate set by dropping
# classes with predicted probability less than self.threshold
class DynamicTopKStrategyDropBase(DynamicTopKStrategy): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyDropBase, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.n_classes = self.args['nClasses']
        self.epsilon = args['k'] # Reviewer: 0.1
        self.threshold = self.epsilon / self.n_classes
        
    def update_k(self): 
        predictions = self.predict(self.train_raw_dataset).cpu().numpy() # (N, n_classes)
        filtered_counts = np.array([np.sum(pred >= self.threshold) for pred in predictions]) # (N,)

        # Assertion on the positiveness of filtered_counts
        assert np.all(filtered_counts > 0), 'filtered_counts should be all positive.'

        self.cur_k = filtered_counts
        self.cur_k = clip_k(self.cur_k, self.n_classes)

        print(self.cur_k)
        print(np.unique(self.cur_k, return_counts=True))
        if self.wandb_run:
            self.wandb_run.log({
                'average_k': np.mean(self.cur_k),
                'cur_k': wandb.Histogram(self.cur_k.tolist())
            })
    
    def update(self, lb_indices):
        self.idxs_lb[lb_indices] = True

class DynamicTopKStrategyRandomDrop(DynamicTopKStrategyDropBase):
    def query(self, n):
        self.update_k()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        chosen_indices = np.random.choice(np.arange(len(idxs_unlabeled)), n, replace=False) if n < len(idxs_unlabeled) else np.arange(len(idxs_unlabeled))

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyEntropyDrop(DynamicTopKStrategyDropBase):
    def query(self, n):
        self.update_k()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=True)

        # sampling N samples
        chosen_indices = entropy_sorted_indices[:n]

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyInvEntropyDrop(DynamicTopKStrategyDropBase):
    def query(self, n):
        self.update_k()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=False)

        # sampling N samples
        chosen_indices = entropy_sorted_indices[:n]

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyBadgeDrop(DynamicTopKStrategyDropBase):
    def query(self, n):
        self.update_k()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        
        probs, embs = self.predict(unlabeled_dataset, return_prob=True, return_embedding=True)
        embs = embs.numpy()
        probs = probs.numpy()

        # the logic below reflects a speedup proposed by Zhang et al.
        # see Appendix D of https://arxiv.org/abs/2306.09910 for more details
        # m = (~self.idxs_lb).sum()
        m = len(idxs_unlabeled)
        mu = None
        D2 = None
        chosen = set()
        chosen_list = []
        emb_norms_square = np.sum(embs ** 2, axis=-1)
        max_inds = np.argmax(probs, axis=-1)

        probs = -1 * probs
        probs[np.arange(m), max_inds] += 1
        prob_norms_square = np.sum(probs ** 2, axis=-1)

        # sampling N samples
        adds = 0
        while adds < len(idxs_unlabeled):
            if len(chosen) >= n: 
                break
            chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2, device=self.device)
            adds += 1
        chosen_indices = np.array(list(chosen_list))

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]
