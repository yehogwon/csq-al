import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import abstractmethod

import os
import gc
import random
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Sampler, Subset
import wandb
from tqdm import tqdm

from time import time
from contextlib import contextmanager

from model.csvm import CSVM

from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
from dataset.common import DatasetWrapper

@contextmanager
def timer(desc: str=''): 
    print(f' ***** Timer Begins ***** ***** {desc} ***** ')
    start = time()
    yield # execute the code
    end = time()
    print(f' ****** Timer Ends ****** ***** {desc} ***** {end - start:.2f} sec')

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

def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# TOOD: check if this works properly with Slurm scheduler
def init_for_distributed(rank, world_size, port, seed): 
    print(f'initializing for distributed: rank ({rank}), world_size ({world_size})')
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://0.0.0.0:{port}', 
        world_size=world_size,
        rank=rank
    )

    set_seed(seed + rank)

    dist.barrier()
    
def mixup(x, y, alpha, device, en_mixup): 
    if not en_mixup:
        return x, y
    else: 
        lam = np.random.beta(alpha, alpha)
        rand_indices = torch.randperm(x.size(0)).to(device)
        x_shuffled = x[rand_indices]
        y_shuffled = y[rand_indices]
        
        x = lam * x + (1 - lam) * x_shuffled
        y_a = y
        y_b = y_shuffled

        return x, y_a, y_b, lam

def mixup_loss(criterion, out, labels_a, labels_b, lam): 
    return lam * criterion(out, labels_a) + (1 - lam) * criterion(out, labels_b)

def _out_train(
    rank: int,
    world_size: int,
    port: int, 
    seed: int, 
    ddp: bool,
    net: torch.nn.Module,
    sync_bn: bool, 
    idxs_train: np.ndarray,
    train_dataset,
    test_dataset,
    tr_batch_size: int,
    tr_num_workers: int,
    te_batch_size: int,
    te_num_workers: int,
    optimizer_name: str,
    scheduler_name: str,
    warmup_epochs: int,
    warmup_lr: float,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    milestones: list, # for MultiStepLR
    gamma: float, # for MultiStepLR
    en_mixup: bool,
    alpha: float, # for mixup
    log_prefix: str,
    wandb_run: object,
    return_dict: dict
):
    if ddp: 
        init_for_distributed(rank, world_size, port, seed)
        net = net.to(rank)
        if sync_bn:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    sampled_train_datset = Subset(train_dataset, idxs_train)

    train_sampler = None
    test_sampler = None
    if ddp: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(sampled_train_datset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    loader_tr = DataLoader(
        sampled_train_datset, 
        sampler=train_sampler, 
        batch_size=tr_batch_size, 
        # batch_size=int(tr_batch_size / world_size),
        num_workers=tr_num_workers,
        shuffle=True if train_sampler is None else False, 
        pin_memory=True,
        persistent_workers=ddp,
    )
    
    loader_te = DataLoader(
        test_dataset,
        sampler=test_sampler,
        # batch_size=int(te_batch_size / world_size),
        batch_size=te_batch_size, 
        num_workers=te_num_workers,
        pin_memory=True,
        persistent_workers=ddp
    )

    if optimizer_name == 'AdamW': 
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else: 
        raise ValueError('Invalid optimizer')
    
    if scheduler_name == 'MultiStepLR': 
        def lr_lambda(epoch): 
            if epoch < warmup_epochs and warmup_epochs > 0:
                return warmup_lr / lr + (1 - warmup_lr / lr) * epoch / warmup_epochs

            new_lr_factor = 1.0
            for milestone in milestones: 
                if epoch >= milestone:
                    new_lr_factor *= gamma
            return new_lr_factor
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        raise ValueError('Invalid scheduler')

    criterion = nn.CrossEntropyLoss()
    
    train_acc = 0.
    best_acc = float('-inf')
    best_top5_acc = 0.
    best_model = None # state dict
    for epoch in range(1, n_epochs + 1):
        # run.py is executed several times when loader is iterated (both in train and val)
        # This was data loader workers!

        # ... train ... #
        net.train()
        if train_sampler is not None: 
            train_sampler.set_epoch(epoch)

        n_corrects = torch.tensor(0.0, device=rank)
        losses = []
        train_iterator = tqdm(enumerate(loader_tr), total=len(loader_tr), desc=f'Train... {log_prefix}epoch{epoch}', leave=False) if rank == 0 else enumerate(loader_tr) # run tqdm only on rank 0
        before = time()
        for idx, (x, y, _) in train_iterator: 
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()

            loss = None
            if en_mixup and (idx + 1) % 3 == 0: 
                x, y_a, y_b, lam = mixup(x, y, alpha, rank, en_mixup)
                out, _ = net(x)
                loss = mixup_loss(criterion, out, y_a, y_b, lam)
            else: 
                out, _ = net(x)
                loss = criterion(out, y)
            loss.backward()

            n_corrects += (torch.argmax(out, 1) == y).sum().item()
            losses.append(loss.item())

            if wandb_run and rank == 0:
                wandb_run.log({
                    log_prefix + 'loss': loss.item()
                })

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        losses = torch.tensor(sum(losses), dtype=torch.float32, device=rank)
        n_batches = torch.tensor(len(loader_tr), dtype=torch.int64, device=rank) # number of batches in this rank

        if ddp:
            dist.all_reduce(n_corrects, op=dist.ReduceOp.SUM)
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
        after = time()

        train_acc = n_corrects.item() / n_batches.item()
        train_loss_avg = losses.item() / n_batches.item()

        if rank == 0: 
            print(f'Train... {log_prefix}epoch{epoch} ** Wall-clock Time: {after - before:.2f}s ** train_acc: {train_acc:.4f}, train_loss_avg: {train_loss_avg:.4f}')

        scheduler.step()

        # ... validate ... #
        n_corrects = torch.tensor(0.0, device=rank)
        n_corrects_top5 = torch.tensor(0.0, device=rank)
        net.eval()
        test_iterator = tqdm(loader_te, total=len(loader_te), desc=f'Val... {log_prefix}epoch{epoch}', leave=False) if rank == 0 else loader_te
        before = time()
        with torch.no_grad():
            for x, y, _ in test_iterator: 
                x, y = x.to(rank), y.to(rank)
                out, _ = net(x)
                prob = F.softmax(out, dim=1)
                pred = prob.argmax(dim=1)
                n_corrects += (pred == y).sum().float()
                n_corrects_top5 += (y.view(-1, 1) == prob.topk(5)[1]).sum().float()
        
        if ddp: 
            dist.all_reduce(n_corrects, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_corrects_top5, op=dist.ReduceOp.SUM)
        after = time()
        
        val_acc = n_corrects.item() / len(loader_te.dataset)
        val_acc_top5 = n_corrects_top5.item() / len(loader_te.dataset)
        
        if rank == 0: 
            print(f'Val... {log_prefix}epoch{epoch} ** Wall-clock Time: {after - before:.2f}s ** val_acc: {val_acc:.4f}, val_acc_top5: {val_acc_top5:.4f}, best_acc: {best_acc:.4f}')
            if best_acc < val_acc: 
                best_acc = val_acc
                best_top5_acc = val_acc_top5

                if isinstance(net, torch.nn.parallel.DistributedDataParallel): 
                    best_model = net.module.state_dict()
                else: 
                    best_model = net.state_dict()
        
        if wandb_run and rank == 0: 
            wandb_run.log({
                log_prefix + 'epoch': epoch,
                log_prefix + 'lr': optimizer.param_groups[0]['lr'],
                log_prefix + 'train_acc': train_acc,
                log_prefix + 'train_loss_avg': train_loss_avg,
                log_prefix + 'val_acc': val_acc, 
                log_prefix + 'val_acc_top5': val_acc_top5
            })

    if ddp: 
        dist.barrier()

    if rank == 0:
        cpu_best_model_dict = {k: v.cpu() for k, v in best_model.items()}
        return_dict['result'] = (best_acc, best_top5_acc, cpu_best_model_dict)
    
    if ddp: 
        dist.barrier()
        dist.destroy_process_group()

def _out_predict(rank, world_size, port, seed, ddp, net, sync_bn, dataset, batch_size, num_workers, return_dict): 
    if ddp: 
        init_for_distributed(rank, world_size, port, seed)
        net = net.to(rank)
        if sync_bn: 
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    
    sampler = None
    if ddp: 
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        sampler = OrderPreservingSampler(dataset, world_size, rank)
    loader = DataLoader(
        dataset, 
        sampler=sampler, 
        shuffle=False, 
        # batch_size=int(batch_size / world_size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    probs = []
    labels = []
    embeddings = []

    net.eval()
    with torch.no_grad(): 
        loader_iterator = tqdm(loader, total=len(loader), desc='Predict...', leave=False) if rank == 0 else loader
        for x, y, _ in loader_iterator:
            x, y = x.to(rank), y.to(rank)
            out, e = net(x)
            prob = F.softmax(out, dim=1)
            
            probs.append(prob.cpu().detach())
            labels.append(y.cpu().detach())
            embeddings.append(e.cpu().detach())

    probs = torch.cat(probs).to(rank)
    labels = torch.cat(labels).to(rank)
    embeddings = torch.cat(embeddings).to(rank)

    assert len(probs) == len(labels) == len(embeddings), f'From rank: {rank}, len(probs): {len(probs)}, len(labels): {len(labels)}, len(embeddings): {len(embeddings)}'
    local_length = len(probs)

    if ddp: 
        gathered_sizes = [torch.zeros(1, dtype=torch.int64, device=rank) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, torch.tensor([local_length], dtype=torch.int64, device=rank))

        gathered_sizes = [int(size.item()) for size in gathered_sizes]
        assert sum(gathered_sizes) == len(dataset), f'From rank: {rank}, sum(gathered_sizes): {sum(gathered_sizes)}, len(dataset): {len(dataset)}'
        
        max_size = max(gathered_sizes)

        prob_padding = torch.zeros(max_size - local_length, probs.size(1), dtype=torch.float32, device=rank)
        label_padding = torch.zeros(max_size - local_length, dtype=torch.int64, device=rank)
        emb_padding = torch.zeros(max_size - local_length, embeddings.size(1), dtype=torch.float32, device=rank)

        probs = torch.cat([probs, prob_padding])
        labels = torch.cat([labels, label_padding])
        embeddings = torch.cat([embeddings, emb_padding])

        gathered_probs = [torch.zeros((max_size, probs.size(1)), dtype=torch.float32, device=rank) for _ in range(world_size)]
        gathered_labels = [torch.zeros((max_size,), dtype=torch.int64, device=rank) for _ in range(world_size)]
        gathered_embeddings = [torch.zeros((max_size, embeddings.size(1)), dtype=torch.float32, device=rank) for _ in range(world_size)]

        dist.all_gather(gathered_probs, probs)
        dist.all_gather(gathered_labels, labels)
        dist.all_gather(gathered_embeddings, embeddings)

        gathered_probs = [t[:size] for t, size in zip(gathered_probs, gathered_sizes)]
        gathered_labels = [t[:size] for t, size in zip(gathered_labels, gathered_sizes)]
        gathered_embeddings = [t[:size] for t, size in zip(gathered_embeddings, gathered_sizes)]
        
        if rank == 0: 
            gathered_probs = [p.cpu().detach() for p in gathered_probs]
            gathered_labels = [l.cpu().detach() for l in gathered_labels]
            gathered_embeddings = [e.cpu().detach() for e in gathered_embeddings]

            probs = torch.cat(gathered_probs)
            labels = torch.cat(gathered_labels)
            embeddings = torch.cat(gathered_embeddings)

            probs = probs[:len(dataset)]
            labels = labels[:len(dataset)]
            embeddings = embeddings[:len(dataset)]
    
    probs = probs.cpu().detach()
    labels = labels.cpu().detach()
    embeddings = embeddings.cpu().detach()
    
    if rank == 0:
        return_dict['result'] = (probs, labels, embeddings)
    
    if ddp:
        dist.barrier()
        dist.destroy_process_group()

def _out_predict_dropout(rank, world_size, port, seed, ddp, net, sync_bn, dataset, n_drops, batch_size, num_workers, return_dict): 
    if ddp: 
        init_for_distributed(rank, world_size, port, seed)
        net = net.to(rank)
        if sync_bn:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    
    sampler = None
    if ddp: 
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(
        dataset, 
        sampler=sampler, 
        shuffle=False, 
        # batch_size=int(batch_size / world_size),
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True
    )

    probs = []
    labels = []
    embeddings = []

    net.train()
    with torch.no_grad(): 
        for dropout_idx in range(1, n_drops + 1):
            in_probs = []
            loader_iterator = tqdm(loader, total=len(loader), desc=f'Predict (Droupout: {dropout_idx}/{n_drops})...', leave=False) if rank == 0 else loader
            for x, y, _ in loader_iterator:
                x, y = x.to(rank), y.to(rank)
                out, e = net(x)
                prob = F.softmax(out, dim=1)
                in_probs.append(prob.cpu().detach())
                labels.append(y.cpu().detach())
                embeddings.append(e.cpu().detach())
            probs.append(torch.cat(in_probs))
    
    labels = torch.cat(labels).to(rank)
    embeddings = torch.cat(embeddings).to(rank)

    probs = torch.stack(probs).to(rank)
    probs = probs.mean(dim=0).to(rank)

    if ddp: 
        gathered_probs = [torch.zeros_like(probs) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels) for _ in range(world_size)]
        gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(world_size)]

        dist.all_gather(gathered_probs, probs)
        dist.all_gather(gathered_labels, labels)
        dist.all_gather(gathered_embeddings, embeddings)
        
        if rank == 0: 
            gathered_probs = [p.cpu().detach() for p in gathered_probs]
            gathered_labels = [l.cpu().detach() for l in gathered_labels]
            gathered_embeddings = [e.cpu().detach() for e in gathered_embeddings]

            probs = torch.cat(gathered_probs)
            labels = torch.cat(gathered_labels)
            embeddings = torch.cat(gathered_embeddings)

            probs = probs[:len(dataset)]
            labels = labels[:len(dataset)]
            embeddings = embeddings[:len(dataset)]
    
    probs = probs.cpu().detach()
    labels = labels.cpu().detach()
    embeddings = embeddings.cpu().detach()
    
    if rank == 0:
        return_dict['result'] = (probs, labels, embeddings)
    
    if ddp:
        dist.barrier()
        dist.destroy_process_group()

class Strategy:
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        self.train_dataset = train_dataset
        self.train_raw_dataset = train_raw_dataset
        self.test_dataset = test_dataset
        self.idxs_lb = idxs_lb
        self.net = net
        self.args = args
        
        self.device = args['device']
        self.n_pool = len(train_dataset)

        self.seed = args['seed']

        self.wandb_run = None
        self.log_prefix = ''

        self.alpha = args['alpha']
        self.en_mixup = self.alpha > 0

        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']

        if self.scheduler == 'CosineAnnealingLR':
            raise ValueError('CosineAnnealingLR is not supported in this version due to warmup')

        self.warmup_epochs = args['warmup_epochs']
        self.warmup_lr = args['warmup_lr']

        if args['push_warmup']: 
            self.args['n_epochs'] += self.warmup_epochs
            self.args['milestones'] = [m + self.warmup_epochs for m in self.args['milestones']] if self.args['milestones'] else []

        self.ddp = self.args['device'] == 'cuda' and torch.cuda.device_count() > 1
        self.world_size = torch.cuda.device_count() if self.ddp else 1

        self.sync_bn = self.ddp and args['sync_bn']
        self.port = args['port']

        if self.ddp:
            new_lr = self.args['lr'] * torch.cuda.device_count()
            print(f'************* DDP ************* lr: {self.args["lr"]} -> {new_lr}')
            self.args['lr'] = new_lr

    @abstractmethod
    def query(self, n):
        pass

    def update(self, lb_indices, **kwargs):
        self.idxs_lb[lb_indices] = True

    def set_log_prefix(self, log_prefix): 
        self.log_prefix = log_prefix

    def train(self): 
        if isinstance(self.net, CSVM):
            self.net.initialize()

            if not isinstance(self.train_dataset.dataset, torch.utils.data.TensorDataset):
                raise TypeError('Dataset must be TensorDataset when with SVM')

            train_set = self.train_dataset.dataset
            test_set = self.test_dataset.dataset

            x_train = train_set.tensors[0][self.idxs_lb].cpu().numpy()
            y_train = train_set.tensors[1][self.idxs_lb].cpu().numpy()

            x_test = test_set.tensors[0].cpu().numpy()
            y_test = test_set.tensors[1].cpu().numpy()

            self.net.fit(x_train, y_train)
            y_pred_prob, _ = self.net(x_test)
            y_pred = y_pred_prob.argmax(axis=1)

            acc = accuracy_score(y_test, y_pred)
            top5_acc = top_k_accuracy_score(y_test, y_pred_prob, k=5)

            # Compute f1 score variations

            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')

            return_dict = {
                'final_accuracy': acc,
                'final_accuracy_top5': top5_acc,
                'final_micro_f1': micro_f1,
                'final_macro_f1': macro_f1,
                'final_weighted_f1': weighted_f1
            }

            return return_dict
        
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        self.net.apply(weight_reset).to(self.device)

        if self.ddp: 
            manager = mp.Manager()
            return_dict = manager.dict()
        else: 
            return_dict = {}

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        tr_batch_size = self.args['loader_tr_args']['batch_size']
        tr_num_workers = self.args['loader_tr_args']['num_workers']
        te_batch_size = self.args['loader_te_args']['batch_size']
        te_num_workers = self.args['loader_te_args']['num_workers']

        if self.ddp:
            mp.spawn(_out_train, args=(
                self.world_size,
                self.port,
                self.seed,
                self.ddp,
                self.net,
                self.sync_bn,
                idxs_train,
                self.train_dataset,
                self.test_dataset,
                tr_batch_size,
                tr_num_workers,
                te_batch_size,
                te_num_workers,
                self.optimizer,
                self.scheduler,
                self.warmup_epochs,
                self.warmup_lr,
                self.args['lr'],
                self.args['weight_decay'],
                self.args['n_epochs'],
                self.args['milestones'],
                self.args['gamma'],
                self.en_mixup,
                self.alpha,
                self.log_prefix,
                self.wandb_run,
                return_dict
            ), nprocs=self.world_size, join=True)
        else:
            _out_train(
                0, 
                1,
                self.port,
                self.seed,
                self.ddp,
                self.net,
                self.sync_bn,
                idxs_train,
                self.train_dataset,
                self.test_dataset,
                tr_batch_size,
                tr_num_workers,
                te_batch_size,
                te_num_workers,
                self.optimizer,
                self.scheduler,
                self.warmup_epochs,
                self.warmup_lr,
                self.args['lr'],
                self.args['weight_decay'],
                self.args['n_epochs'],
                self.args['milestones'],
                self.args['gamma'],
                self.en_mixup,
                self.alpha,
                self.log_prefix,
                self.wandb_run,
                return_dict
            )
        
        if 'result' not in return_dict: 
            raise ValueError('result not in return_dict')
        
        best_acc, best_top5_acc, cpu_net_dict = return_dict['result']
        self.net.load_state_dict(cpu_net_dict)
        self.net = self.net.to(self.device)

        return_dict = {
            'final_accuracy': best_acc,
            'final_accuracy_top5': best_top5_acc
        }

        return return_dict
    
    def stop_condition(self) -> bool: 
        return sum(~self.idxs_lb) == 0

    def predict(self, dataset: DatasetWrapper, return_prob=True, return_label=False, return_embedding=False): 
        if not return_prob and not return_label and not return_embedding: 
            raise ValueError('At least one of return_prob, return_label, return_embedding should be True')
        
        if isinstance(self.net, CSVM):
            if return_embedding: 
                raise ValueError('CSVM does not support return_embedding')

            x = np.array([dataset[i][0] for i in range(len(dataset))])
            y = torch.tensor([dataset[i][1] for i in range(len(dataset))]).cpu()
            
            pred_prob_np, _ = self.net(x)
            pred_prob = torch.tensor(pred_prob_np).cpu()

            if return_prob and return_label:
                return pred_prob, y
            elif return_prob:
                return pred_prob
            elif return_label:
                return y

        # """
        if self.ddp:
            manager = mp.Manager()
            return_dict = manager.dict()
            mp.spawn(_out_predict, args=(
                self.world_size,
                self.port,
                self.seed,
                self.ddp,
                self.net,
                self.sync_bn,
                dataset,
                self.args['loader_te_args']['batch_size'],
                self.args['loader_te_args']['num_workers'],
                return_dict
            ), nprocs=self.world_size, join=True)
        else:
            return_dict = {}
            _out_predict(
                0,
                1,
                self.port,
                self.seed,
                self.ddp,
                self.net,
                self.sync_bn,
                dataset,
                self.args['loader_te_args']['batch_size'],
                self.args['loader_te_args']['num_workers'],
                return_dict
            )
        
        if 'result' not in return_dict:
            raise ValueError('result not in return_dict')
        
        probs, labels, embeddings = return_dict['result']
        # """

        """
        loader = DataLoader(
            dataset,
            batch_size=self.args['loader_te_args']['batch_size'],
            num_workers=self.args['loader_te_args']['num_workers']
        )
        
        probs = []
        labels = []
        embeddings = []

        self.net.eval()
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e = self.net(x)
                prob = F.softmax(out, dim=1)
                probs.append(prob.cpu().detach())
                labels.append(y.cpu().detach())
                embeddings.append(e.cpu().detach())
        
        probs = torch.cat(probs)
        labels = torch.cat(labels)
        embeddings = torch.cat(embeddings)
        """

        returns = []
        if return_prob: 
            returns.append(probs)
        if return_label: 
            returns.append(labels)
        if return_embedding:
            returns.append(embeddings)
        
        if len(returns) == 1: 
            return returns[0]
        else: 
            return tuple(returns)

    def predict_prob_dropout(self, dataset, n_drop, return_prob=True, return_label=False, return_embedding=False):
        raise NotImplementedError('predict_prob_dropout is deprecated.')
    
        if isinstance(self.net, CSVM):
            raise ValueError('CSVM does not support predict_prob_dropout')

        if not return_prob and not return_label and not return_embedding: 
            raise ValueError('At least one of return_prob, return_label, return_embedding should be True')
        
        if self.ddp:
            manager = mp.Manager()
            return_dict = manager.dict()
            mp.spawn(_out_predict_dropout, args=(
                self.world_size,
                self.port,
                self.seed,
                self.ddp,
                self.net,
                self.sync_bn,
                dataset,
                n_drop,
                self.args['loader_te_args']['batch_size'],
                self.args['loader_te_args']['num_workers'],
                return_dict
            ), nprocs=self.world_size, join=True)
        else:
            return_dict = {}
            _out_predict_dropout(
                0,
                1,
                self.port,
                self.seed,
                self.ddp,
                self.net,
                self.sync_bn,
                dataset,
                n_drop,
                self.args['loader_te_args']['batch_size'],
                self.args['loader_te_args']['num_workers'],
                return_dict
            )

        if 'result' not in return_dict:
            raise ValueError('result not in return_dict')
        
        probs, labels, embeddings = return_dict['result']

        returns = []
        if return_prob: 
            returns.append(probs)
        if return_label: 
            returns.append(labels)
        if return_embedding:
            returns.append(embeddings)
        
        if len(returns) == 1: 
            return returns[0]
        else: 
            return tuple(returns)
