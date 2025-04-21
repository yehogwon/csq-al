import numpy as np
import torch
from .strategy import Strategy
from .strategy import init_for_distributed, OrderPreservingSampler

import copy
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset, DataLoader

from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats

from tqdm import tqdm

def init_centers(X, K):
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
            D2 = np.minimum(D2, newD)
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: 
            breakpoint()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X_array[ind])
        indsAll.append(ind)
        cent += 1
    return np.array(indsAll)

def _out_get_max_perturbed_loss(
    rank: int, 
    world_size: int, 
    port: int,
    seed: int,
    ddp: bool, 
    net: list,
    dataset,
    saal_batch_size: int,
    num_workers: int,
    rho: float,
    return_dict: dict
):
    net = copy.deepcopy(net).to(rank)
    if ddp: 
        init_for_distributed(rank, world_size, port, seed)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    
    sampler = None
    if ddp: 
        sampler = OrderPreservingSampler(dataset, world_size, rank)
    
    dataloader = DataLoader(dataset, sampler=sampler, shuffle=False, batch_size=saal_batch_size, num_workers=num_workers, pin_memory=True)

    max_perturbed_loss = []
    for x, _, __ in tqdm(dataloader, desc='Maximally perturbed loss'):
        x = x.to(rank)
        out, _ = net(x)
        loss = F.cross_entropy(out, out.argmax(dim=1), reduction='none')
        loss.mean().backward()
        
        norm = torch.norm(
            torch.stack([(torch.abs(p)*p.grad).norm(p=2) for p in net.parameters()]), 
            p=2
        )
        scale = rho / (norm + 1e-12)

        with torch.no_grad(): 
            param_copies = []
            for p in net.parameters():
                param_copies.append(p.clone())
                e_w = (torch.pow(p, 2)) * p.grad * scale.to(p)
                p.add_(e_w)
        
        output_updated, _ = net(x)
        loss_updated = F.cross_entropy(output_updated, out.argmax(dim=1), reduction='none')
        max_perturbed_loss.append(loss_updated.detach().to(rank))
        
        with torch.no_grad():
            for p, p_orig in zip(net.parameters(), param_copies):
                p.copy_(p_orig)
        
        del out, loss, param_copies, e_w, output_updated
    
    max_perturbed_loss = torch.cat(max_perturbed_loss, dim=0)
    local_length = max_perturbed_loss.size(0)

    if ddp: 
        gathered_sizes = [torch.zeros(1, dtype=torch.int64, device=rank) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, torch.tensor([local_length], dtype=torch.int64, device=rank))

        gathered_sizes = [int(size.item()) for size in gathered_sizes]
        assert sum(gathered_sizes) == len(dataset), f'From rank: {rank}, sum(gathered_sizes): {sum(gathered_sizes)}, len(dataset): {len(dataset)}'
        
        max_size = max(gathered_sizes)

        padding_length = max_size - local_length
        if padding_length > 0: 
            padding = torch.zeros(padding_length, dtype=torch.float32, device=rank)
            max_perturbed_loss = torch.cat([max_perturbed_loss, padding])

        gathered_max_perturbed_loss = [torch.zeros((max_size,), dtype=torch.float32, device=rank) for _ in range(world_size)]
        dist.all_gather(gathered_max_perturbed_loss, max_perturbed_loss)

        gathered_max_perturbed_loss = [t[:size] for t, size in zip(gathered_max_perturbed_loss, gathered_sizes)]

        if rank == 0: 
            gathered_max_perturbed_loss = [t.cpu().detach() for t in gathered_max_perturbed_loss]
            max_perturbed_loss = torch.cat(gathered_max_perturbed_loss)
            max_perturbed_loss = max_perturbed_loss[:len(dataset)]
    
    if rank == 0: 
        max_perturbed_loss = max_perturbed_loss.cpu().detach()
        return_dict['max_perturbed_loss'] = max_perturbed_loss
    
    if ddp: 
        dist.barrier()
        dist.destroy_process_group()

def get_max_perturbed_loss(net, dataset, rho: float, saal_batch_size: int, num_workers: int, ddp: bool, world_size: int, port: int, seed: int):
    if ddp: 
        manager = mp.Manager()
        return_dict = manager.dict()
        mp.spawn(_out_get_max_perturbed_loss, args=(
            world_size, 
            port, 
            seed, 
            ddp, 
            net, 
            dataset, 
            saal_batch_size, 
            num_workers, 
            rho, 
            return_dict
        ), nprocs=world_size, join=True)
    else: 
        return_dict = {}
        _out_get_max_perturbed_loss(
            0, 
            1, 
            port, 
            seed, 
            ddp, 
            net, 
            dataset, 
            saal_batch_size, 
            num_workers, 
            rho, 
            return_dict
        )
    
    if 'max_perturbed_loss' not in return_dict:
        raise ValueError('max_perturbed_loss not in return_dict')
    
    return return_dict['max_perturbed_loss']

class SAAL(Strategy):
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        super(SAAL, self).__init__(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        
        self.rho = args['rho']
        self.saal_batch_size = args['saal_batch_size']
        self.diversity = args['diversity']
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        dataset = Subset(self.train_raw_dataset, idxs_unlabeled)

        # This ensures that there are enough samples to query
        if len(idxs_unlabeled) <= n: 
            return idxs_unlabeled
        
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
            chosen = init_centers(max_perturbed_loss, n)
            chosen = np.array(chosen, dtype=int)
        else: 
            chosen = max_perturbed_loss.sort(descending=True)[1][:n]
        
        return idxs_unlabeled[chosen]
