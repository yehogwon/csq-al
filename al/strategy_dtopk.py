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

from .badge_sampling import init_centers as init_centers_badge
from .badge_sampling import distance
from .prob_cover import construct_graph
from .saal import init_centers as init_centers_saal
from .saal import get_max_perturbed_loss

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

class DynamicTopKStrategy(Strategy): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategy, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.n_classes = self.args['nClasses']
        self.k = args['k']
        self.cur_k = np.array([self.n_classes] * len(train_raw_dataset))

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
        self.update_k()
    
    def compute_cost(self, target_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
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

        cur_costs = in_tf * in_cost + out_tf * double_cost # (N)
        
        costs.append(cur_costs)
        gt_index_list.append(gt_indices)
        in_indices += (gt_indices_onebase <= k).tolist()

        costs = torch.cat(costs)
        gt_indices = torch.cat(gt_index_list)

        in_indices = torch.tensor(in_indices)
        in_ratio = torch.sum(in_indices).item() / len(in_indices)

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

class DynamicTopKStrategySAAL(DynamicTopKStrategy):
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
            max_perturbed_loss = get_max_perturbed_loss(
                self.net, 
                dataset, 
                self.rho, 
                self.saal_batch_size, 
                self.args['loader_te_args']['num_workers'],
                self.ddp, 
                self.world_size, 
                self.port, 
                self.seed
            )

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
    
    def dummy_query(self, q_idxs): 
        """r
        This function skips querying process and directly returns the indices of the samples to be labeled.
        The argument q_idxs is the indices of the samples to be labeled.
        This function for computing and analyzing the annotation for the queried samples.
        """
        
        # Note: q_idxs is in the indexing space of the full training set.
        # Also, all the indices in this function are in the same indexing space.

        # choose K samples from N samples and label them, using as calibration set
        assert len(q_idxs) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(q_idxs)), self.args['calibration_set_size'], replace=False)
        calibration_indices = q_idxs[calibration_indices_in_chosen_indices]
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, calibration_indices)

        calib_costs, _, _ = self.compute_cost(calibration_indices)
        calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(q_idxs)
        mask = torch.ones(len(q_idxs), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        print(f'{self.log_prefix} ::: total_cost: {total_cost}')

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost
            })

        # return
        return q_idxs

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, _gt_indices, _in_ratio = self.compute_cost(in_ratio_indices=chosen_indices)

        unlabeled_k = torch.from_numpy(self.cur_k[idxs_unlabeled]).to(self.device)
        pred_cost = self.get_predcost(unlabeled_k)
        
        acquisition = 1 / (pred_cost + 1)
        acquisition_sorted_list = torch.argsort(acquisition, descending=True)
        
        # sampling N samples
        chosen_indices = acquisition_sorted_list[:n]

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(chosen)
        mask = torch.ones(len(chosen), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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

class DynamicTopKStrategySAALConf(DynamicTopKStrategyConfBase):
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
            max_perturbed_loss = get_max_perturbed_loss(
                self.net,
                dataset,
                self.rho,
                self.saal_batch_size,
                self.args['loader_te_args']['num_workers'],
                self.ddp,
                self.world_size,
                self.port,
                self.seed
            )

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
        costs, gt_indices, in_ratio = self.compute_cost(chosen)
        mask = torch.ones(len(chosen), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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

class DynamicTopKStrategyHybridSAALConf(DynamicTopKStrategyConfBase): 
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
            max_perturbed_loss = get_max_perturbed_loss(
                self.net, 
                dataset, 
                self.rho, 
                self.saal_batch_size, 
                self.args['loader_te_args']['num_workers'],
                self.ddp,
                self.world_size,
                self.port,
                self.seed
            )

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
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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

############################# Adaptive Alpha Cost Logging #############################
class DynamicTopKStrategyEntropyConfAdapLog(DynamicTopKStrategyConfBase):
    def update_k(self): 
        calib_probs, calib_labels = self.predict(self.conf_calibration_dataset, return_prob=True, return_label=True)
        except_probs, except_labels = self.predict(self.sampled_minus_calib_datset, return_prob=True, return_label=True)
        calib_scores = self.calculate_nonconformity_scores(calib_probs, calib_labels)
        n_cal = self.args['calibration_set_size']

        if self.adaptive_epsilon: 
            clipped_log2 = lambda x: torch.log2(torch.clip(x, min=1, max=self.n_classes).float())

            epsilon_candidates = np.linspace(0, 0.98, 50)
            epsilon_cost_table = {}
            epsilon_cost_table_for_except = {}
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

                ############### For actively sampled dataset - except calibration dataset ###############
                set_sizes = torch.sum(1 - except_probs <= q, dim=1)
                arg_sort_probs = torch.argsort(except_probs, dim=1, descending=True)

                costs = clipped_log2(set_sizes + 1)
                for i, label in enumerate(except_labels):
                    if label not in arg_sort_probs[i, :set_sizes[i]]:
                        costs[i] += clipped_log2(self.n_classes - set_sizes[i])
                
                total_cost_except = torch.sum(costs).item()
                epsilon_cost_table_for_except[alpha] = total_cost_except
            
            best_epsilon = min(epsilon_cost_table, key=epsilon_cost_table.get)
            
            print('************************************** **************************************')
            print(f'Epsilon-Cost Table: {epsilon_cost_table}')
            print(f'Best Epsilon: {best_epsilon}')
            print('************************************** **************************************')
            print('********* For Actively Sampled Dataset - Except Calibration Dataset *********')
            print(f'Epsilon-Cost Table for Except: {epsilon_cost_table_for_except}')
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
        self.cur_k = clip_k(self.cur_k, self.n_classes)

        print(self.cur_k)
        print(np.unique(self.cur_k, return_counts=True))
        if self.wandb_run:
            self.wandb_run.log({
                'average_k': np.mean(self.cur_k),
                'cur_k': wandb.Histogram(self.cur_k.tolist())
            })

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        probs = self.predict(unlabeled_dataset)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=True)

        # sampling N samples
        chosen_indices = entropy_sorted_indices[:n]

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
        calib_cost = torch.sum(calib_costs).item()

        # find the indices of actively sampled data - calibration set, and obtain them with index space of unlabeled dataset
        chosen_indices_arange = np.arange(len(chosen_indices))
        except_indices_chosen = chosen_indices_arange[~np.isin(chosen_indices_arange, calibration_indices_in_chosen_indices)]
        except_indices = chosen_indices[except_indices_chosen] # in unlabeled
        except_indices_train = idxs_unlabeled[except_indices] # in train
        self.sampled_minus_calib_datset = Subset(self.train_raw_dataset, except_indices_train)

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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

############################# Wall Clock Time Measurement (LEGACY) #############################
class DynamicTopKStrategyEntropyWall(DynamicTopKStrategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyEntropyWall, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

        if self.ddp: 
            raise ValueError('DynamicTopKStrategyRandomConfWall does not support DDP do separate dataloder')

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        unlabeled_loader = DataLoader(unlabeled_dataset, shuffle=False, **self.args['loader_te_args'])
        probs = []
        self.net.eval()
        
        start = time()
        with torch.no_grad():
            for x, y, _ in unlabeled_loader: 
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
                prob = F.softmax(out, dim=1)
                probs.append(prob.cpu().detach())
        probs = torch.cat(probs)
        entropies = -torch.sum(probs * torch.log(probs), dim=1)
        entropy_sorted_indices = torch.argsort(entropies, descending=True)
        
        end = time()
        print(f' ************************ Entropy Sampling Wall Clock Time: {end - start} (sec) ************************ ')
        exit(0)
        
        """
        costs, gt_indices, in_ratio = self.compute_cost()
        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio
            })
        
        ordered_costs = costs[entropy_sorted_indices]
        cost_cumsum = torch.cumsum(ordered_costs, dim=0)
        if max(cost_cumsum) <= n: 
            chosen_indices = entropy_sorted_indices
        else: 
            idx = torch.where(cost_cumsum > n)[0][0].item()
            chosen_indices = entropy_sorted_indices[:idx]
        chosen_indices = chosen_indices.cpu().numpy()

        total_round_cost = torch.sum(costs[chosen_indices])
        if self.wandb_run:
            self.wandb_run.log({
                'total_round_cost': total_round_cost.item()
            })
        """
        
        return idxs_unlabeled[chosen_indices]

class DynamicTopKStrategyBadgeWall(DynamicTopKStrategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyBadgeWall, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

        if self.ddp: 
            raise ValueError('DynamicTopKStrategyRandomConfWall does not support DDP do separate dataloder')

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        
        """
        costs, gt_indices, in_ratio = self.compute_cost()
        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()), 
                'in_ratio': in_ratio
            })
        """
        
        unlabeled_loader = DataLoader(unlabeled_dataset, shuffle=False, **self.args['loader_te_args'])
        self.net.eval()

        start = time()

        embedding = []
        probs = []
        with torch.no_grad():
            for x, y, _ in unlabeled_loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.net(x)
                embedding.append(e1.detach().cpu())
                pr = F.softmax(out,1)
                probs.append(pr.detach().cpu())
        embedding = torch.cat(embedding)
        probs = torch.cat(probs)

        embs = embedding.numpy()
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
            chosen, chosen_list, mu, D2 = init_centers_badge((probs, prob_norms_square), (embs, emb_norms_square), chosen, chosen_list, mu, D2, device=self.device)
            adds += 1
        
        end = time()
        print(f' ************************ BADGE Sampling Wall Clock Time: {end - start} (sec) ************************ ')
        exit(0)

        """
        total_round_cost = torch.sum(rec_cost[chosen_list])
        if self.wandb_run:
            self.wandb_run.log({
                'total_round_cost': total_round_cost.item()
            })
        """
        
        return idxs_unlabeled[chosen_list]

class DynamicTopKStrategyRandomConfWall(DynamicTopKStrategy): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyRandomConfWall, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

        if self.ddp: 
            raise ValueError('DynamicTopKStrategyRandomConfWall does not support DDP do separate dataloder')

        self.n_classes = self.args['nClasses']
        self.epsilon = args['k']

        assert self.epsilon > 0, 'epsilon should be greater than 0'
        
        self.conf_calibration_dataset = None
    
    def calculate_nonconformity_scores(self, probs, true_labels):
        return 1 - torch.gather(probs, 1, true_labels.view(-1, 1)).squeeze()
        
    def update_k(self, sample_indices): 
        self.net.eval()

        calib_loader = DataLoader(self.conf_calibration_dataset, shuffle=False, **self.args['loader_te_args'])
        sampled_loader = DataLoader(Subset(self.train_raw_dataset, sample_indices), shuffle=False, **self.args['loader_te_args'])
        
        start = time()

        # The inference steps below are implemented without using predict() method
        # to remove the overhead caused by declaring loaders
        calib_probs, calib_labels = [], []
        for x, y, _ in calib_loader:
            with torch.no_grad():
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
                prob = F.softmax(out, dim=1)
                calib_probs.append(prob)
                calib_labels.append(y)
        calib_probs = torch.cat(calib_probs)
        calib_labels = torch.cat(calib_labels)
        calib_scores = self.calculate_nonconformity_scores(calib_probs, calib_labels)

        n_cal = self.args['calibration_set_size']
        adjusted_threshold = compute_threshold(self.epsilon, n_cal)
        alpha = float(np.quantile(calib_scores.cpu().numpy(), adjusted_threshold, method='higher'))

        probs = []
        with torch.no_grad():
            for x, y, _ in sampled_loader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
                prob = F.softmax(out, dim=1)
                probs.append(prob)
        probs = torch.cat(probs)
        nonconformity = 1 - probs
        predictions = [torch.where(nc <= alpha)[0] for nc in nonconformity]

        self.cur_k = np.array([len(pred) for pred in predictions])
        self.cur_k = clip_k(self.cur_k, self.n_classes)

        end = time()
        print(f' ************************ Conformal Pred. Wall Clock Time: {end - start} (sec) ************************ ')
        exit(0)

        print(self.cur_k)
        print(np.unique(self.cur_k, return_counts=True))
        if self.wandb_run:
            self.wandb_run.log({
                'average_k': np.mean(self.cur_k),
                'cur_k': wandb.Histogram(self.cur_k.tolist())
            })
    
    def update(self, lb_indices):
        self.idxs_lb[lb_indices] = True
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_dataset = Subset(self.train_raw_dataset, idxs_unlabeled)
        entropies = torch.rand(len(unlabeled_dataset))
        entropy_sorted_indices = torch.argsort(entropies, descending=True)

        # sampling N samples
        chosen_indices = entropy_sorted_indices[:n]

        # choose K samples from N samples and label them, using as calibration set
        assert len(chosen_indices) >= self.args['calibration_set_size'], 'The number of labeled samples should be not less than the desired calibration set size'
        calibration_indices_in_chosen_indices = np.random.choice(np.arange(len(chosen_indices)), self.args['calibration_set_size'], replace=False)
        calibration_indices = chosen_indices[calibration_indices_in_chosen_indices] # in unlabeled
        self.conf_calibration_dataset = Subset(self.train_raw_dataset, idxs_unlabeled[calibration_indices])

        calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
        calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

        # log artifacts
        total_cost = calib_cost + other_cost

        if self.wandb_run:
            self.wandb_run.log({
                'gt_indices': wandb.Histogram(gt_indices.cpu().tolist()),
                'in_ratio': in_ratio,
                'total_round_cost': total_cost.item(),
            })

        # return
        return idxs_unlabeled[chosen_indices]

############################# Wall Clock Time Measurement (NEW) #############################
class DynamicTopKStrategyHybridEntropyConfAdapWall(DynamicTopKStrategy): 
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(DynamicTopKStrategyHybridEntropyConfAdapWall, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        self.n_classes = self.args['nClasses']
        self.epsilon = args['k']
        self.d = args['d']

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
        _start = time()

        calib_probs, calib_labels = self.predict(self.conf_calibration_dataset, return_prob=True, return_label=True)
        calib_scores = self.calculate_nonconformity_scores(calib_probs, calib_labels)

        if self.adaptive_epsilon: 
            clipped_log2 = lambda x: torch.log2(torch.clip(x, min=1, max=self.n_classes).float())

            epsilon_candidates = np.linspace(0, 0.98, 50)
            epsilon_cost_table = {}
            for alpha in tqdm(epsilon_candidates, desc='Searching alpha (epsilon)'): 
                alpha = alpha.item()
                q = float(np.quantile(calib_scores.cpu().numpy(), 1 - alpha, method='higher')) # Q(alpha)

                set_sizes = torch.sum(1 - calib_probs <= q, dim=1) # (K,)
                arg_sort_probs = torch.argsort(calib_probs, dim=1, descending=True) # (K, C)

                costs = clipped_log2(set_sizes + 1) # (K,)
                for i, label in enumerate(calib_labels):
                    if label not in arg_sort_probs[i, :set_sizes[i]]: 
                        costs[i] += clipped_log2(self.n_classes - set_sizes[i])
                total_cost = torch.sum(costs).item()
                
                epsilon_cost_table[alpha] = total_cost
            
            best_epsilon = min(epsilon_cost_table, key=epsilon_cost_table.get)
            self.epsilon = best_epsilon

            if self.wandb_run:
                self.wandb_run.log({
                    'epsilon': self.epsilon
                })

        n_cal = self.args['calibration_set_size']
        adjusted_threshold = compute_threshold(self.epsilon, n_cal)
        q = float(np.quantile(calib_scores.cpu().numpy(), adjusted_threshold, method='higher')) # Q(alpha)

        unlabeled_dataset = Subset(self.train_raw_dataset, np.arange(self.n_pool)[~self.idxs_lb])
        predictions = self.conformal_prediction(unlabeled_dataset, q)

        _end = time()
        print(f'********** Wall-clock Time for Adaptive-Alpha with adaptive set to {self.adaptive_epsilon}: {_end - _start} (sec) **********')

        # self.cur_k = np.array([len(pred) for pred in predictions])
        # self.cur_k = clip_k(self.cur_k, self.n_classes)
        self.cur_k = np.random.randint(1, self.n_classes + 1, len(self.train_dataset))

        print(self.cur_k)
        print(np.unique(self.cur_k, return_counts=True))
        if self.wandb_run:
            self.wandb_run.log({
                'average_k': np.mean(self.cur_k),
                'cur_k': wandb.Histogram(self.cur_k.tolist())
            })
    
    def update(self, lb_indices):
        self.idxs_lb[lb_indices] = True
    
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

        calib_costs, _, _ = self.compute_cost(idxs_unlabeled[calibration_indices])
        calib_cost = torch.sum(calib_costs).item()

        # update Q(alpha)
        self.update_k()

        # label N-K samples
        costs, gt_indices, in_ratio = self.compute_cost(idxs_unlabeled[chosen_indices])
        mask = torch.ones(len(chosen_indices), dtype=torch.bool)
        mask[calibration_indices_in_chosen_indices] = False
        other_cost = torch.sum(costs[mask]).item()

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
