import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

class OrderPreservingSampler(Sampler): 
    def __init__(self, dataset, world_size: int, rank: int): 
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

        dataset_size = len(dataset)
        samples_per_rank = dataset_size // world_size
        leftover = dataset_size % world_size

        if rank < leftover:
            self.num_samples_this_rank = samples_per_rank + 1
            self.start_idx = rank * (samples_per_rank + 1)
        else:
            self.num_samples_this_rank = samples_per_rank
            self.start_idx = rank * samples_per_rank + leftover

        self.end_idx = self.start_idx + self.num_samples_this_rank

    def __len__(self): 
        return self.num_samples_this_rank
    
    def __iter__(self): 
        return iter(range(self.start_idx, self.end_idx))

class DistributedWeightedRandomSampler(Sampler): 
    pass
