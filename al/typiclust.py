'''
This implementation is adopted from the GitHub repository: 
    https://github.com/avihu111/TypiClust
[1] Guy Hacohen, Avihu Dekel, Daphna Weinshall
    Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets. ICML 2022
'''

import os
from typing import Union

import numpy as np
import pandas as pd
from .strategy import Strategy

from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans, KMeans

import torch
import faiss

def _get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)

    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]

def _get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = _get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance

def calculate_typicality(features, num_neighbors):
    mean_distance = _get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality

def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_

class TypiClust(Strategy):
    MIN_CLUSTER_SIZE = 5
    N_CLUSTERS = 500
    K_NN = 20

    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        raise NotImplementedError('TypiClust does not guarantee its proper working')
        super(TypiClust, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

        self.features_path = args['features_path']

        if not os.path.exists(self.features_path):
            raise ValueError(f'No such file: {self.features_path}')

        if self.features_path.endswith('.npy'):
            self.features = np.load(self.features_path)
        elif self.features_path.endswith('.pth'):
            self.features = torch.load(self.features_path).cpu().numpy()
        else: 
            raise ValueError(f'Unsupported features file format: {self.features_path}')

        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True) # (N, 512)
        self.clusters = kmeans(self.features, num_clusters=self.N_CLUSTERS)

    def query(self, n):
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        if len(idxs_unlabeled) <= n: 
            return idxs_unlabeled

        clusters = np.copy(self.clusters)

        cluster_ids, cluster_sizes = np.unique(clusters, return_counts=True)
        cluster_labeled_counts = np.bincount(clusters[idxs_labeled], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({
            'cluster_id': cluster_ids,
            'cluster_size': cluster_sizes,
            'existing_count': cluster_labeled_counts,
            'neg_cluster_size': -1 * cluster_sizes
        })

        # Omit too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]

        # Sort clusters by the number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        
        clusters[idxs_labeled] = -1

        chosen = []

        # no clusters or all clusters are labeled (all -1's)
        if len(clusters_df) == 0:
            return np.random.choice(idxs_unlabeled, n, replace=False)
        
        for i in tqdm(range(n), desc='TypiClust'): 
            if clusters.max() == -1: 
                break
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (clusters == cluster).nonzero()[0]
            rel_feats = self.features[indices]
            breakpoint()
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            chosen.append(idx)
            clusters[idx] = -1
        
        if len(chosen) < n:
            print(' *** WARNING: Not enough samples to choose -> Randomly choose the rest')
            remaining = np.setdiff1d(idxs_unlabeled, chosen)
            chosen.extend(np.random.choice(remaining, n - len(chosen), replace=False))
        
        return np.array(chosen)
