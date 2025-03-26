'''
This implementation is adopted from the GitHub repository: 
    https://github.com/avihu111/TypiClust
[1] Ofer Yehuda, Avihu Dekel, Guy Hacohen, Daphna Weinshall
    Active Learning Through a Covering Lens. NeurIPS 2022
'''

import os
from typing import Union

import numpy as np
import pandas as pd
from .strategy import Strategy

from tqdm import tqdm

import torch

def construct_graph(features: np.ndarray, delta: int, batch_size: int=128) -> pd.DataFrame:
    cuda_feats = torch.tensor(features).cuda()
    edges = [] # (x, y, d) x -> y with distance d. x and y are indices in the full training set
    xs, ys, ds = [], [], []
    for i in tqdm(range(len(features) // batch_size), desc='Constructing graph for ProbCover'): 
        cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(cur_feats, cuda_feats) # (batch_size, N)
        mask = dist < delta # B_delta(x)
        x, y = mask.nonzero().T # pairs of indices whose distances are less than delta but non-zero

        xs.append(x.cpu() + batch_size * i)
        ys.append(y.cpu())
        ds.append(dist[mask].cpu())
    
    xs = torch.cat(xs).cpu().numpy()
    ys = torch.cat(ys).cpu().numpy()
    ds = torch.cat(ds).cpu().numpy()
    
    edge_df = pd.DataFrame({'source': xs, 'target': ys, 'distance': ds})

    print(f'Information of the graph with delta={delta}')
    print('  - Number of nodes:', len(features))
    print('  - Number of edges:', len(edge_df))

    return edge_df

class ProbCover(Strategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(ProbCover, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)

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

            # new_covered_samples = edge_df['target'][edge_df['source'] == node].values
            # edge_df = edge_df[~np.isin(edge_df['target'], new_covered_samples)]

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
        return np.array(chosen)
