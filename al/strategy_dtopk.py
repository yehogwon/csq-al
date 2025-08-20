from abc import abstractmethod
from typing import Tuple, Optional, List, Union

import random
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

from .badge_sampling import init_centers as init_centers_badge
from .badge_sampling import distance
from .prob_cover import construct_graph
from .saal import init_centers as init_centers_saal
from .saal import SAAL

from scipy import stats
from sklearn.metrics import pairwise_distances
import pandas as pd

import os

from time import time

def compute_threshold(epsilon, n_cal): 
    threshold = 1 - epsilon
    # threshold = math.ceil((1 - threshold) * (n_cal + 1)) / n_cal # LEGACY
    threshold = np.clip(threshold, 0, 1)
    return threshold

def clip_k(k, n_classes): 
    k = np.where(k < 1, n_classes, k)
    k = np.clip(k, 1, n_classes)
    return k

def _is_empty(l: Union[List, np.ndarray, torch.Tensor]) -> bool:
    if isinstance(l, list):
        return len(l) == 0
    elif isinstance(l, np.ndarray):
        return l.size == 0
    elif isinstance(l, torch.Tensor):
        return l.numel() == 0
    else:
        raise TypeError(f'Unsupported type: {type(l)}')

class DynamicTopKStrategy(Strategy): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategy, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.n_classes = self.args['nClasses']
        self.k = args['k']
        self.cur_k = np.array([self.n_classes] * len(train_raw_dataset))

        self.use_partial_labels: bool = args['partial']
        if self.use_partial_labels and self.en_mixup:
            raise ValueError(
                'Mixup cannot be used with CE and NL at the same time'
            )
        assert not (self.use_partial_labels and self.en_mixup), 'Mixup cannot be used with partial labels'

        if self.use_partial_labels: 
            r"""
            Below is applicable only for labeled samples,
            which can be examined with `self.idxs_lb`.
            For `self.labels`, 0 indicates "not that class" while
            1 indicates "possibly that class".
            In the beginning, all classes are possible,
            so initially set to all one.
            """
            # TODO: Store in a spare array for memory efficiency
            self.partial_labels = np.ones((self.n_pool, self.n_classes), dtype=bool)

        self.oracle_wrong_rate = args['oracle_wrong_rate']
        self.wrong_within_candidate = args['wrong_within_candidate']
        assert self.oracle_wrong_rate >= 0 and self.oracle_wrong_rate <= 1, 'oracle_wrong_rate should be a probability'

    def adaptive_k(self) -> list[int]: 
        probs = self.predict(self.train_raw_dataset) # (N, C)
        counts = []
        for prob in probs: 
            # prob: (C)
            sorted_prob, _ = torch.sort(prob, descending=True)
            cumsum = torch.cumsum(sorted_prob, dim=0)
            count = torch.where(cumsum > self.k)[0][0].item() + 1
            counts.append(count)
            del sorted_prob, cumsum
        counts = np.array(counts)
        return counts
    
    def get_k(self, idx: int) -> int: # not used
        if self.k >= 1: 
            return self.cur_k
        else: 
            return self.cur_k[idx]
    
    def update_k(self): 
        self.cur_k = self.k if self.k >= 1 else self.adaptive_k()
        self.cur_k = np.array(self.cur_k)
        print(self.cur_k)
        print(np.unique(self.cur_k, return_counts=True))
    
    def update(self, lb_indices):
        self.idxs_lb[lb_indices] = True

        # partial labels (initial pool)
        if hasattr(self, 'partial_labels') and \
            self.partial_labels[lb_indices].all(): 
            for idx in lb_indices:
                _gt = self.train_dataset[idx][1]
                self.partial_labels[idx] = 0
                self.partial_labels[idx][_gt] = 1

        self.update_k()
    
    @property
    def full_label_idxs(self):
        _sums = self.partial_labels.sum(axis=1)  # (N,)
        assert (_sums > 0).all(), 'Samples with no possible label exist'
        indices = np.where(_sums == 1)[0]
        return indices
    
    @property
    def partial_label_idxs(self):
        return np.setdiff1d(np.arange(self.n_pool), self.full_label_idxs)
    
    def _update_partial_labels(
        self,
        idxs_of_interest: np.ndarray,
        probs: torch.Tensor,
        labels: torch.Tensor,
        set_sizes: torch.Tensor,
        in_tf: torch.Tensor
    ):
        labels_np = labels.cpu().numpy()
        in_tf_np = in_tf.cpu().numpy()
        out_tf_np = ~in_tf_np

        if (self.partial_labels[idxs_of_interest].sum(axis=1) == 1).any():
            raise ValueError('Samples with full labels chosen again')

        idxs_of_in = idxs_of_interest[in_tf_np]  # int array
        idxs_of_out = idxs_of_interest[out_tf_np]  # int array

        # full labels
        _len_in = len(idxs_of_in)
        _onehots = np.zeros((_len_in, self.n_classes), dtype=bool)
        _onehots[np.arange(_len_in), labels_np[in_tf_np]] = 1
        self.partial_labels[idxs_of_in] = _onehots

        # partial labels
        out_probs_np = probs.cpu().numpy()[out_tf_np]
        out_set_sizes = set_sizes.cpu().numpy()[out_tf_np]
        # TODO: Use array parallelism
        for dataset_index, prob, set_size in zip(idxs_of_out, out_probs_np, out_set_sizes): 
            topk_preds = np.argsort(prob)[-set_size:]
            _labels = self.partial_labels[dataset_index]  # (C,)
            _labels[topk_preds] = False
            self.partial_labels[dataset_index] = _labels
    
    def _flip_labels_in_candidates(
        self,
        idxs_of_interest: np.ndarray,
        sorted_prob_indices: torch.Tensor,
        set_sizes: torch.Tensor
    ):
        flip_mask = torch.rand(len(idxs_of_interest)) <= self.oracle_wrong_rate
        local_flip_indices = torch.where(flip_mask)[0]  # long tensor
        for i, lcl_flip_idx in enumerate(local_flip_indices):
            dataset_index = idxs_of_interest[lcl_flip_idx]
            candidate_set = sorted_prob_indices[i][:set_sizes[i]]
            remaining_set = sorted_prob_indices[i][set_sizes[i]:]

            if len(candidate_set) == self.n_classes or self.wrong_within_candidate:
                # candidate set contains all labels
                new_idx_in_cs = random.randint(0, len(candidate_set) - 1)
                new_label = candidate_set[new_idx_in_cs].item()
            else:
                new_idx_in_cs = random.randint(0, len(candidate_set))
                if new_idx_in_cs == len(candidate_set):  # none of above
                    gt_label = self.train_raw_dataset.get_label(dataset_index)
                    if gt_label in remaining_set:
                        new_label = gt_label
                    else:
                        new_idxs_out_cs = random.randint(0, len(remaining_set) - 1)
                        new_label = remaining_set[new_idxs_out_cs].item()
                else:
                    new_label = candidate_set[new_idx_in_cs].item()

            self.train_dataset.set_label(dataset_index, new_label)
    
    def compute_cost(
        self,
        target_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # length of target_indices should be positive
        assert len(target_indices) > 0, 'target_indices should be provided as a non-empty list of indices'
        
        # target_indices: indices to compute cost, in_ratio, gt_indices (indexing space: training set)
        k = torch.tensor(self.cur_k)
        if len(k.size()) == 0: # 0d
            k = k.repeat(self.n_pool)
        
        dataset = Subset(self.train_raw_dataset, target_indices)
        k = k[target_indices]

        costs = []
        gt_index_list = []
        in_indices = [] # True: included / False: not included
        
        probs, labels = self.predict(dataset, return_prob=True, return_label=True)
        sorted_prob_indices = torch.sort(probs, dim=1, descending=True)[1] # (N, C), sorted indices
        gt_indices = (sorted_prob_indices == labels.unsqueeze(1)).nonzero()[:, 1] # (N,)
        gt_indices_onebase = gt_indices + 1

        in_tf = (gt_indices_onebase <= k).float() # (N), 1: included
        out_tf = (gt_indices_onebase > k).float() # (N), 1: not included

        clipped_log2 = lambda x: torch.log2(torch.clip(x, min=1, max=self.n_classes).float())

        in_cost = clipped_log2(k + 1) # (N)
        out_cost = clipped_log2(self.n_classes - k) # (N)
        # out_cost[torch.isinf(out_cost)] = 0
        double_cost = in_cost + out_cost

        if self.use_partial_labels:  # only one query
            cur_costs = in_cost # (N)
        else:  # two queries only if needed
            cur_costs = in_tf * in_cost + out_tf * double_cost # (N)
        
        costs.append(cur_costs)
        gt_index_list.append(gt_indices)
        in_indices += (gt_indices_onebase <= k).tolist()

        costs = torch.cat(costs)
        gt_indices = torch.cat(gt_index_list)

        in_indices = torch.tensor(in_indices)
        in_ratio = torch.sum(in_indices).item() / len(in_indices)

        if self.use_partial_labels:
            self._update_partial_labels(
                target_indices,
                probs,
                labels,
                k,
                in_tf.bool()
            )

        if self.oracle_wrong_rate > 0:
            self._flip_labels_in_candidates(
                target_indices,
                sorted_prob_indices,
                k,
            )

        return costs, gt_indices, in_ratio

    @abstractmethod
    def query(self, n):
        pass

class DynamicTopKStrategyRandom(DynamicTopKStrategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyRandom, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        
        chosen_indices = np.random.choice(np.arange(len(idxs_unlabeled)), n, replace=False) if n < len(idxs_unlabeled) else np.arange(len(idxs_unlabeled))

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_round_cost = torch.sum(costs).item()
        
        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio,
                'total_round_cost': total_round_cost
            })
        
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyEntropy(DynamicTopKStrategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyEntropy, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=True)
        
        chosen_indices = entropy_sorted_indices[:n]

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_round_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio,
                'total_round_cost': total_round_cost
            })
        
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyCoreset(DynamicTopKStrategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyCoreset, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        lb_flag = self.idxs_lb.copy()
        embedding = self.predict(self.train_raw_dataset, return_prob=False, return_embedding=True)
        embedding = embedding.numpy()
        
        ##### Furthest First #####
        X = embedding[idxs_unlabeled, :]
        X_set = embedding[lb_flag, :]
        
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        chosen_indices = []

        for _ in range(len(idxs_unlabeled)):
            if len(chosen_indices) >= n:
                break
            idx = min_dist.argmax()
            chosen_indices.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_round_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio,
                'total_round_cost': total_round_cost
            })

        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyBadge(DynamicTopKStrategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyBadge, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.deterministic = args['deterministic']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        if len(idxs_unlabeled) < n:
            chosen_list = np.arange(len(idxs_unlabeled))
        else: 
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

            adds = 0
            while adds < len(idxs_unlabeled):
                if len(chosen) >= n: 
                    break
                chosen, chosen_list, mu, D2 = init_centers_badge((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2, device=self.device, deterministic=self.deterministic)
                adds += 1

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_list])
        total_round_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio,
                'total_round_cost': total_round_cost
            })
        
        return idxs_unlabeled[chosen_list]

class DynamicTopKStrategyProbCover(DynamicTopKStrategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyProbCover, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

        self.features_path = args['features_path']

        if not os.path.exists(self.features_path):
            raise ValueError(f'No such file: {self.features_path}')

        if self.features_path.endswith('.npy'):
            self.features = np.load(self.features_path)
        elif self.features_path.endswith('.pth'):
            self.features = torch.load(self.features_path)
        else: 
            raise ValueError(f'Unsupported features file format: {self.features_path}')

        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True) # (N, 512)

        self.delta: float = self.args['delta']
        self.edge_df: pd.DataFrame = construct_graph(self.features, self.delta, batch_size=500)

    def query(self, n):
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        covered_samples_by_labeled = np.unique(self.edge_df[self.edge_df['source'].isin(idxs_labeled)]['target'])
        edge_df = self.edge_df[~self.edge_df['target'].isin(covered_samples_by_labeled)]

        chosen = [] # indexing space: full training set

        samples_to_choose = min(n, len(idxs_unlabeled))
        pbar = tqdm(range(samples_to_choose), desc='Querying ProbCover')
        for _ in pbar:
            if len(edge_df) == 0:
                pbar.update(samples_to_choose - len(chosen))
                break
            degrees = np.bincount(edge_df['source'], minlength=len(self.features))
            node = np.argmax(degrees)
            pbar.set_description(f'Querying ProbCover: {len(chosen)}/{samples_to_choose} :: Node {node}')

            new_covered_samples = np.unique(edge_df[edge_df['source'] == node]['target'])
            edge_df = edge_df[~edge_df['target'].isin(new_covered_samples)]

            if node in chosen or node in idxs_labeled: 
                breakpoint()
                raise RuntimeError(f'Node {node} is already chosen or labeled')
            
            chosen.append(node)
        
        if len(chosen) < samples_to_choose:
            print(' *** WARNING: Not enough samples to choose -> Randomly choose the rest')
            remaining = np.setdiff1d(idxs_unlabeled, chosen)
            chosen.extend(np.random.choice(remaining, samples_to_choose - len(chosen), replace=False))
        
        chosen = np.array(chosen)

        costs, gt_indices, in_ratio = self.compute_cost(chosen)
        total_round_cost = torch.sum(costs).item()
        
        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio,
                'total_round_cost': total_round_cost
            })
        
        return chosen

class DynamicTopKStrategySAAL(DynamicTopKStrategy, SAAL):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategySAAL, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.rho = args['rho']
        self.diversity = args['diversity']
        self.saal_batch_size = args['saal_batch_size']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        dataset = Subset(self.train_raw_dataset, idxs_unlabeled)

        # This ensures that there are enough samples to query
        if len(idxs_unlabeled) <= n: 
            chosen_indices = np.arange(len(idxs_unlabeled))
        else: 
            max_perturbed_loss = self.get_max_perturbed_loss(idxs_unlabeled)

            if self.diversity: 
                chosen_indices = init_centers_saal(max_perturbed_loss, n)
                chosen_indices = np.array(chosen_indices, dtype=int)
            else: 
                chosen_indices = max_perturbed_loss.sort(descending=True)[1][:n]

        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        total_round_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio,
                'total_round_cost': total_round_cost
            })
        
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyConfBase(DynamicTopKStrategy): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyConfBase, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.n_classes = self.args['nClasses']
        self.epsilon = args['k']

        self.adaptive_epsilon = self.epsilon == 0
        self.conf_calibration_dataset = None
    
    def get_predcost(self, predk: torch.Tensor) -> torch.Tensor: 
        ### get adjusted alpha
        n_cal = self.args['calibration_set_size']
        adjusted_threshold = compute_threshold(self.epsilon, n_cal)

        ### calcualte cost
        clipped_log2 = lambda x: torch.log2(torch.clip(x, min=1, max=self.n_classes).float())
        hit_cost = clipped_log2(predk.float() + 1)
        miss_cost = clipped_log2(predk.float() + 1) + clipped_log2(float(self.args['nClasses']) - predk.float())

        pred_cost = adjusted_threshold * hit_cost + (1 - adjusted_threshold) * miss_cost

        return pred_cost
    
    def calculate_nonconformity_scores(self, probs, true_labels):
        return 1 - torch.gather(probs, 1, true_labels.view(-1, 1)).squeeze()

    def conformal_prediction(self, dataset, threshold):
        probs = self.predict(dataset)
        nonconformity = 1 - probs
        return [torch.where(nc <= threshold)[0] for nc in nonconformity]
        
    def update_k(self): 
        calib_probs, calib_labels = self.predict(self.conf_calibration_dataset, return_prob=True, return_label=True)
        calib_scores = self.calculate_nonconformity_scores(calib_probs, calib_labels)
        n_cal = self.args['calibration_set_size']

        if self.adaptive_epsilon: 
            clipped_log2 = lambda x: torch.log2(torch.clip(x, min=1, max=self.n_classes).float())

            epsilon_candidates = np.linspace(0, 0.98, 50)
            epsilon_cost_table = {}
            for alpha in tqdm(epsilon_candidates, desc='Searching alpha (epsilon)'): 
                alpha = alpha.item()
                q = float(np.quantile(calib_scores.cpu().numpy(), compute_threshold(alpha, n_cal), method='higher')) # Q(alpha)

                set_sizes = torch.sum(1 - calib_probs <= q, dim=1) # (K,)
                arg_sort_probs = torch.argsort(calib_probs, dim=1, descending=True) # (K, C)

                costs = clipped_log2(set_sizes + 1) # (K,)
                in_count = 0
                for i, label in enumerate(calib_labels):
                    if label not in arg_sort_probs[i, :set_sizes[i]]: 
                        costs[i] += clipped_log2(self.n_classes - set_sizes[i])
                    else:
                        in_count += 1

                in_ratio = in_count / len(calib_labels)
                total_cost = torch.sum(costs).item()
                
                epsilon_cost_table[alpha] = total_cost
            
            best_epsilon = min(epsilon_cost_table, key=epsilon_cost_table.get)
            
            print('************************************** **************************************')
            print(f'Epsilon-Cost Table: {epsilon_cost_table}')
            print(f'Best Epsilon: {best_epsilon}')
            print('************************************** **************************************')

            self.epsilon = best_epsilon

            if self.wandb_run:
                self.wandb_run.log({
                    'epsilon': self.epsilon
                })

        adjusted_threshold = compute_threshold(self.epsilon, n_cal)
        q = float(np.quantile(calib_scores.cpu().numpy(), adjusted_threshold, method='higher')) # Q(alpha)
        predictions = self.conformal_prediction(self.train_raw_dataset, q)

        self.cur_k = np.array([len(pred) for pred in predictions])

        if self.args['verbose']:
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            unlabeled_k = self.cur_k[idxs_unlabeled]

            print(' ******************** VERBOSE BEGINS ******************** ')
            print(' Statistics of self.cur_k')
            print('  - min:', np.min(self.cur_k))
            print('  - max:', np.max(self.cur_k))
            print('  - avg:', np.mean(self.cur_k))
            print('  - out of [1, n_classes]:', np.sum(self.cur_k < 1), np.sum(self.cur_k > self.n_classes))
            print('  - zero:', np.sum(self.cur_k == 0))
            print('    * Note: cur_k should be nonnegative since it is the length of a list.')
            print(' Statistics of self.cur_k[unlabeled]')
            if len(unlabeled_k) == 0:
                print('  - It is empty ;(')
            else: 
                print('  - len:', len(unlabeled_k))
                print('  - min:', np.min(unlabeled_k))
                print('  - max:', np.max(unlabeled_k))
                print('  - avg:', np.mean(unlabeled_k))
                print('  - out of [1, n_classes]:', np.sum(unlabeled_k < 1), np.sum(unlabeled_k > self.n_classes))
                print('  - zero:', np.sum(unlabeled_k == 0))
                print('    * Note: cur_k should be nonnegative since it is the length of a list.')
            print(' ********************* VERBOSE ENDS ********************* ')

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

class DynamicTopKStrategyRandomConf(DynamicTopKStrategyConfBase):
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)

        # sampling N samples
        chosen_indices = np.random.choice(np.arange(len(idxs_unlabeled)), n, replace=False) if n < len(idxs_unlabeled) else np.arange(len(idxs_unlabeled))

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyEntropyConf(DynamicTopKStrategyConfBase):
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=True)

        # sampling N samples
        chosen_indices = entropy_sorted_indices[:n].cpu().numpy()

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyInvEntropyConf(DynamicTopKStrategyConfBase):
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=False)

        # sampling N samples
        chosen_indices = entropy_sorted_indices[:n].cpu().numpy()

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyCostConf(DynamicTopKStrategyConfBase):
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)

        unlabeled_k = torch.from_numpy(self.cur_k[idxs_unlabeled]).to(self.device)
        pred_cost = self.get_predcost(unlabeled_k)
        
        acquisition = 1 / (pred_cost + 1)
        acquisition_sorted_list = torch.argsort(acquisition, descending=True)
        
        # sampling N samples
        chosen_indices = acquisition_sorted_list[:n].cpu().numpy()

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyCoresetConf(DynamicTopKStrategyConfBase):
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        lb_flag = self.idxs_lb.copy()
        embedding = self.predict(self.train_raw_dataset, return_prob=False, return_embedding=True)
        embedding = embedding.numpy()

        # sampling N samples
        ##### Furthest First #####
        X = embedding[idxs_unlabeled, :]
        X_set = embedding[lb_flag, :]
        
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        chosen_indices = []
        for _ in range(len(idxs_unlabeled)):
            if len(chosen_indices) >= n:
                break
            idx = min_dist.argmax()
            chosen_indices.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
        
        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })
        
        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyBadgeConf(DynamicTopKStrategyConfBase):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyBadgeConf, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.deterministic = args['deterministic']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        if len(idxs_unlabeled) < n:
            chosen_list = np.arange(len(idxs_unlabeled))
        else: 
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
                chosen, chosen_list, mu, D2 = init_centers_badge((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2, device=self.device, deterministic=self.deterministic)
                adds += 1
        chosen_indices = np.array(list(chosen_list))

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyProbCoverConf(DynamicTopKStrategyConfBase):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyProbCoverConf, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

        self.features_path = args['features_path']

        if not os.path.exists(self.features_path):
            raise ValueError(f'No such file: {self.features_path}')

        if self.features_path.endswith('.npy'):
            self.features = np.load(self.features_path)
        elif self.features_path.endswith('.pth'):
            self.features = torch.load(self.features_path)
        else: 
            raise ValueError(f'Unsupported features file format: {self.features_path}')

        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True) # (N, 512)

        self.delta: float = self.args['delta']
        self.edge_df: pd.DataFrame = construct_graph(self.features, self.delta, batch_size=500)

    def query(self, n):
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        covered_samples_by_labeled = np.unique(self.edge_df[self.edge_df['source'].isin(idxs_labeled)]['target'])
        edge_df = self.edge_df[~self.edge_df['target'].isin(covered_samples_by_labeled)]

        chosen = [] # indexing space: full training set

        samples_to_choose = min(n, len(idxs_unlabeled))
        pbar = tqdm(range(samples_to_choose), desc='Querying ProbCover')
        for _ in pbar:
            if len(edge_df) == 0:
                pbar.update(samples_to_choose - len(chosen))
                break
            degrees = np.bincount(edge_df['source'], minlength=len(self.features))
            node = np.argmax(degrees)
            pbar.set_description(f'Querying ProbCover: {len(chosen)}/{samples_to_choose} :: Node {node}')

            new_covered_samples = np.unique(edge_df[edge_df['source'] == node]['target'])
            edge_df = edge_df[~edge_df['target'].isin(new_covered_samples)]

            if node in chosen or node in idxs_labeled: 
                breakpoint()
                raise RuntimeError(f'Node {node} is already chosen or labeled')
            
            chosen.append(node)
        
        if len(chosen) < samples_to_choose:
            print(' *** WARNING: Not enough samples to choose -> Randomly choose the rest')
            remaining = np.setdiff1d(idxs_unlabeled, chosen)
            chosen.extend(np.random.choice(remaining, samples_to_choose - len(chosen), replace=False))
        
        chosen = np.array(chosen)
        
        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen[calibration_indices_in_chosen_indices] # in the entire training set
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, calibration_indices)

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(calibration_indices)
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            np.setdiff1d(chosen, calibration_indices)
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })
        
        # return
        return chosen

class DynamicTopKStrategySAALConf(DynamicTopKStrategyConfBase, SAAL):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategySAALConf, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.rho = args['rho']
        self.saal_batch_size = args['saal_batch_size']
        self.diversity = args['diversity']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        dataset = Subset(self.train_raw_dataset, idxs_unlabeled)

        # sampling N samples
        if len(idxs_unlabeled) <= n: 
            chosen_indices = np.arange(len(idxs_unlabeled))
        else: 
            max_perturbed_loss = self.get_max_perturbed_loss(dataset)

            if self.diversity: 
                chosen_indices = init_centers_saal(max_perturbed_loss, n)
                chosen_indices = np.array(chosen_indices, dtype=int)
            else: 
                chosen_indices = max_perturbed_loss.sort(descending=True)[1][:n]

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyHybridEntropyConf(DynamicTopKStrategyConfBase): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyHybridEntropyConf, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.d = args['d']
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset) # (U,) <- size of unlabeled dataset, this is on CPU
        entropies = -torch.sum(probs * torch.log(probs), dim=1) # (U,) on CPU
        
        # sampling N samples
        unlabeled_k = torch.from_numpy(self.cur_k[idxs_unlabeled]).to(self.device) # (U,)
        pred_cost = self.get_predcost(unlabeled_k)
        pred_cost_np = pred_cost.cpu().numpy()

        acquisition = np.power(1 + entropies, self.d) / pred_cost_np # (N,)
        acquisition_sorted_indices = torch.argsort(acquisition, descending=True)
        
        chosen_indices = acquisition_sorted_indices[:n]

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyHybridBadgeConf(DynamicTopKStrategyConfBase): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyHybridBadgeConf, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.d = args['d']
        self.deterministic = args['deterministic']

    def init_centers(self, X1, X2, chosen, chosen_list,  mu, D2, costs: np.ndarray, device='cpu', deterministic=False):
        if len(chosen) == 0:
            ind = np.argmax(X1[1] * X2[1])
            mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
            D2 = distance(X1, X2, mu[0]).ravel().astype(float)
            D2[ind] = 0
        else:
            newD = distance(X1, X2, mu[-1]).ravel().astype(float)
            D2 = np.minimum(D2, newD)
            D2[chosen_list] = 0
            Ddist: np.ndarray = (D2 ** 2) / np.sum(D2 ** 2)

            Ddist = np.nan_to_num(Ddist, nan=0.0, posinf=0.0, neginf=0.0)
            Ddist /= np.sum(Ddist)

            dist = np.power(1 + Ddist, self.d)
            dist /= costs
            dist /= sum(dist)
            dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
            dist /= sum(dist)

            if deterministic: 
                sorted_dist_indices = np.argsort(dist)
                added = False
                for i in sorted_dist_indices:
                    if i not in chosen:
                        ind = i
                        added = True
                        break
                if not added: 
                    raise ValueError('No sample to add')
            else:
                customDist = stats.rv_discrete(name='custm', values=(np.arange(len(dist)), dist))
                ind = customDist.rvs(size=1)[0]
                while ind in chosen: ind = customDist.rvs(size=1)[0]
            mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
        chosen.add(ind)
        chosen_list.append(ind)
        return chosen, chosen_list, mu, D2
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        if len(idxs_unlabeled) < n:
            chosen_list = np.arange(len(idxs_unlabeled))
        else: 
            unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
            
            unlabeled_k = torch.from_numpy(self.cur_k[idxs_unlabeled]).to(self.device)
            pred_cost = self.get_predcost(unlabeled_k)
            pred_cost_np = pred_cost.cpu().numpy()

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
                chosen, chosen_list, mu, D2 = self.init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2, pred_cost_np, device=self.device, deterministic=self.deterministic)
                adds += 1
        chosen_indices = np.array(list(chosen_list))

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyHybridProbCoverConf(DynamicTopKStrategyConfBase): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyHybridProbCoverConf, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.d = args['d']

        self.features_path = args['features_path']

        if not os.path.exists(self.features_path):
            raise ValueError(f'No such file: {self.features_path}')

        if self.features_path.endswith('.npy'):
            self.features = np.load(self.features_path)
        elif self.features_path.endswith('.pth'):
            self.features = torch.load(self.features_path)
        else: 
            raise ValueError(f'Unsupported features file format: {self.features_path}')

        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True) # (N, 512)

        self.delta: float = self.args['delta']
        self.edge_df: pd.DataFrame = construct_graph(self.features, self.delta, batch_size=500)

    def query(self, n):
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        k = torch.from_numpy(self.cur_k).to(self.device)
        pred_cost = self.get_predcost(k)
        pred_cost_np = pred_cost.cpu().numpy()

        covered_samples_by_labeled = np.unique(self.edge_df[self.edge_df['source'].isin(idxs_labeled)]['target'])
        edge_df = self.edge_df[~self.edge_df['target'].isin(covered_samples_by_labeled)]

        chosen = [] # indexing space: full training set
        
        samples_to_choose = min(n, len(idxs_unlabeled))
        pbar = tqdm(range(samples_to_choose), desc='Querying ProbCover')
        for _ in pbar:
            if len(edge_df) == 0:
                pbar.update(samples_to_choose - len(chosen))
                break
            degrees = np.bincount(edge_df['source'], minlength=len(self.features))
            normalized_degrees = degrees / np.max(degrees)

            acquisition = np.power(1 + normalized_degrees, self.d) / pred_cost_np
            arg_sorted_acquisition = np.argsort(-acquisition) # descending order

            arg_sorted_acquisition_without_labeled = arg_sorted_acquisition[~np.isin(arg_sorted_acquisition, np.concatenate([idxs_labeled, chosen]))]
            node = arg_sorted_acquisition_without_labeled[0]

            pbar.set_description(f'Querying ProbCover: {len(chosen)}/{samples_to_choose} :: Node {node}')

            new_covered_samples = np.unique(edge_df[edge_df['source'] == node]['target'])
            edge_df = edge_df[~edge_df['target'].isin(new_covered_samples)]

            if node in chosen or node in idxs_labeled: 
                breakpoint()
                raise RuntimeError(f'Node {node} is already chosen or labeled')
            
            chosen.append(node)
        
        if len(chosen) < samples_to_choose:
            print(' *** WARNING: Not enough samples to choose -> Cost-efficiently choose the rest')
            remaining = np.setdiff1d(idxs_unlabeled, chosen)
            # pick the samples with low pred_cost_np in remaining
            remaining_pred_cost = pred_cost_np[remaining]
            remaining_sorted_indices = np.argsort(-remaining_pred_cost)
            remaining_sorted = remaining[remaining_sorted_indices]
            chosen.extend(remaining_sorted[:samples_to_choose - len(chosen)])
        
        chosen = np.array(chosen)
        
        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen[calibration_indices_in_chosen_indices] # in the entire training set
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, calibration_indices)

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(calibration_indices)
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            np.setdiff1d(chosen, calibration_indices)
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })
        
        # return
        return chosen

class DynamicTopKStrategyHybridSAALConf(DynamicTopKStrategyConfBase, SAAL): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyHybridSAALConf, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.rho = args['rho']
        self.saal_batch_size = args['saal_batch_size']
        self.diversity = args['diversity']
        self.d = args['d']

    def init_centers(self, X, K, costs): 
        X_array = np.expand_dims(X, 1)
        ind = np.argmax([np.linalg.norm(s, 2) for s in X_array])    # s should be array-like.
        mu = [X_array[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        # print('#Samps\tTotal Distance')
        for _ in tqdm(range(K - len(mu)), desc='k-means++ initialization'):
            if len(mu) == 1:
                D2 = pairwise_distances(X_array, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X_array, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: 
                breakpoint()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)

            dist = np.power(1 + Ddist, self.d)
            dist /= costs
            dist /= sum(dist)
            dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
            dist /= sum(dist)

            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(dist)), dist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X_array[ind])
            indsAll.append(ind)
            cent += 1
        return np.array(indsAll)
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        
        # sampling N samples
        unlabeled_k = torch.from_numpy(self.cur_k[idxs_unlabeled]).to(self.device) # (U,)
        pred_cost = self.get_predcost(unlabeled_k)
        pred_cost_np = pred_cost.cpu().numpy()

        if len(idxs_unlabeled) <= n: 
            chosen_indices = np.arange(len(idxs_unlabeled))
        else: 
            max_perturbed_loss = self.get_max_perturbed_loss(dataset)

            if self.diversity: 
                chosen_indices = self.init_centers(max_perturbed_loss, n, pred_cost_np)
            else: 
                acquisition = np.power(1 + max_perturbed_loss, self.d) / pred_cost_np
                acquisition_sorted_indices = torch.argsort(acquisition, descending=True)
                chosen_indices = acquisition_sorted_indices[:n]

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        if self.args['cq_calib']: 
            calib_cost = np.log2(self.n_classes) * self.args['calibration_set_size']
        else: 
            calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
            calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(
            idxs_unlabeled[np.setdiff1d(chosen_indices, calibration_indices)]
        )
        other_cost = torch.sum(costs).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return idxs_unlabeled[chosen_indices]

class UBDynamicTopKStrategyRandom(Strategy): # Upper Bound, Random
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(UBDynamicTopKStrategyRandom, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        chosen_indices = np.random.choice(np.arange(len(idxs_unlabeled)), n, replace=False) if n < len(idxs_unlabeled) else np.arange(len(idxs_unlabeled))
        sampled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[chosen_indices])

        probs, y = self.predict(sampled_dataset, return_prob=True, return_label=True)
        prob_sorted_indices = torch.argsort(probs, dim=1, descending=True).cpu() # (N, C)

        gt_indices = (prob_sorted_indices == y.unsqueeze(1)).nonzero()[:, 1] # (N,)
        costs = torch.log2(gt_indices + 1)
        total_round_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'total_round_cost': total_round_cost
            })
        
        return idxs_unlabeled[chosen_indices]

class UBDynamicTopKStrategyEntropy(Strategy): # Upper Bound, Entropy
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(UBDynamicTopKStrategyEntropy, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=True)
        
        chosen_indices = entropy_sorted_indices[:n]

        sampled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[chosen_indices])

        y_list = []
        y_train_dataloader = DataLoader(sampled_dataset, shuffle=False, **self.args['loader_te_args'])
        for x, y, _ in tqdm(y_train_dataloader, desc='Gathering Y\'s'):
            y_list.append(y.cpu())
        y = torch.cat(y_list).cpu()

        sampled_probs = probs[chosen_indices]
        sampled_prob_sorted_indices = torch.argsort(sampled_probs, dim=1, descending=True).cpu()

        gt_indices = (sampled_prob_sorted_indices == y.unsqueeze(1)).nonzero()[:, 1] # (N,)
        costs = torch.log2(gt_indices + 1) # (N,)
        
        total_round_cost = torch.sum(costs).item()

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'total_round_cost': total_round_cost
            })
        
        return idxs_unlabeled[chosen_indices]

class UBDynamicTopKStrategyBadge(Strategy): # Upper Bound, BADGE
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        raise NotImplementedError('UBDynamicTopKStrategyBadge is no longer supported.')
        super(UBDynamicTopKStrategyBadge, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

    def init_centers(self, X1, X2, chosen, chosen_list,  mu, D2, device='cpu'):
        if len(chosen) == 0:
            ind = np.argmax(X1[1] * X2[1])
            mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
            D2 = distance(X1, X2, mu[0]).ravel().astype(float)
            D2[ind] = 0
        else:
            newD = distance(X1, X2, mu[-1]).ravel().astype(float)
            D2 = np.minimum(D2, newD)
            D2[chosen_list] = 0
            Ddist: np.ndarray = (D2 ** 2) / sum(D2 ** 2)
            Ddist = Ddist / sum(Ddist)

            # Debugging and validation
            if not np.isclose(np.sum(Ddist), 1.0):
                raise ValueError(f'The sum of provided pk is not 1: {np.sum(Ddist)} | {np.isnan(Ddist).any()} | {np.isinf(Ddist).any()} | {np.min(Ddist)} | {np.max(Ddist)}')

            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in chosen: ind = customDist.rvs(size=1)[0]
            mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
        chosen.add(ind)
        chosen_list.append(ind)
        return chosen, chosen_list, mu, D2
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)

        y_list = []
        y_train_dataloader = DataLoader(Subset(self.train_raw_dataset, idxs_unlabeled), shuffle=False, **self.args['loader_te_args'])
        for x, y, _ in tqdm(y_train_dataloader, desc='Gathering Y\'s'):
            y_list.append(y.cpu())
        y = torch.cat(y_list).cpu()

        unlabeled_k = torch.from_numpy(self.cur_k[idxs_unlabeled]).to(self.device)

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

        adds = 0
        while adds < len(idxs_unlabeled):
            if len(chosen) >= n: 
                break
            chosen, chosen_list, mu, D2 = self.init_centers((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2, device=self.device)
            adds += 1

        sampled_probs = probs[chosen_list]
        sampled_prob_sorted_indices = torch.argsort(sampled_probs, dim=1, descending=True).cpu()

        gt_indices = (sampled_prob_sorted_indices == y.unsqueeze(1)).nonzero()[:, 1]
        costs = torch.log2(gt_indices + 1)

        total_round_cost = torch.sum(costs[chosen_list]).item()
        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'total_round_cost': total_round_cost
            })

        return idxs_unlabeled[chosen_list]
