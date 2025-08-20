import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import abstractmethod
from typing import Callable

import os
import gc
import tempfile
import math
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

import time
from contextlib import contextmanager

from model.csvm import CSVM

from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
from dataset.common import DatasetWrapper
from dataset.sampler import OrderPreservingSampler, DistributedWeightedRandomSampler

# NOTE: Do not use this since it causes danling child processes
# torch.multiprocessing.set_sharing_strategy('file_system')

EPS = 1e-10

@contextmanager
def timer(desc: str=''): 
    print(f' ***** Timer Begins ***** ***** {desc} ***** ')
    start = time.time()
    yield # execute the code
    end = time.time()
    print(f' ****** Timer Ends ****** ***** {desc} ***** {end - start:.2f} sec')

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
        init_method=f'tcp://localhost:{port}', 
        world_size=world_size,
        rank=rank
    )

    set_seed(seed + rank)
    dist.barrier()

def _save_ckpt(model_dict: dict) -> str:
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        fpath = tmp_file.name
    torch.save(model_dict, fpath)
    print(f'Saved best model to {fpath} : waiting for a second')
    time.sleep(1)
    return fpath

def _load_ckpt(fpath: str, remove: bool=True) -> dict:
    ckpt = torch.load(fpath, map_location='cpu', weights_only=True)
    if remove: _remove_ckpt(fpath)
    return ckpt

def _remove_ckpt(fpath: str):
    if os.path.exists(fpath):
        os.remove(fpath)

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
    net_constructor: Callable,
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
    steps: int, # for ExponentialLR
    gamma: float, # for MultiStepLR and ExponentialLR
    en_mixup: bool,
    alpha: float, # for mixup
    log_prefix: str,
    wandb_run: object,
    return_dict: dict
):
    net = net_constructor().to(rank)
    if ddp: 
        print(f"[Rank {rank}] Initializing distributed process group...", flush=True)
        init_for_distributed(rank, world_size, port, seed)
        print(f"[Rank {rank}] Distributed process group initialized", flush=True)
        if sync_bn:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
        print(f"[Rank {rank}] DDP model initialized", flush=True)

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
    elif optimizer_name == 'RMSProp': 
        optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
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
    elif scheduler_name == 'ExponentialLR':
        def lr_lambda(epoch): 
            if epoch < warmup_epochs and warmup_epochs > 0:
                return warmup_lr / lr + (1 - warmup_lr / lr) * epoch / warmup_epochs

            hot_epoch = max(0, epoch - warmup_epochs)
            new_lr_factor = 1.0
            _decay_count = hot_epoch // steps
            new_lr_factor *= gamma ** _decay_count
            return new_lr_factor
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == 'CosineAnnealingLR':
        def lr_lambda(epoch):
            if epoch < warmup_epochs and warmup_epochs > 0:
                return warmup_lr / lr + (1 - warmup_lr / lr) * epoch / warmup_epochs
            
            hot_epoch = max(0, epoch - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * hot_epoch / (n_epochs - warmup_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError('Invalid scheduler')

    criterion = nn.CrossEntropyLoss()
    
    train_acc = 0.
    best_acc = float('-inf')
    best_top5_acc = 0.
    best_model = None # state dict

    if hasattr(train_dataset, 'partial_labels'):
        if hasattr(train_dataset, 'partial_loss') and len(train_dataset.partial_loss) > 0:
            partial_criteria = []

            def _negative_loss(partial_probs: torch.Tensor, partial_labels: torch.Tensor):
                negative_labels = 1 - partial_labels  # (p, C), 1 only for NOT GT classes
                negative_loss_tensor = -negative_labels * torch.log2(1 - partial_probs + EPS)  # (p, C)
                # Directly applying mean unstabilizes the training
                # since the indicator makes no use of some samples
                # negative_loss = torch.sum(negative_loss_tensor * indicator, dim=1).mean()
                indicator = (negative_labels * partial_probs) >= (1 / partial_labels.shape[1])  # (p, C)
                _negative_loss = torch.sum(negative_loss_tensor * indicator, dim=1)  # (p,)
                negative_loss = _negative_loss / torch.sum(negative_labels, dim=1)

                samplewise_indicator = torch.sum(indicator, dim=1) > 0  # Binary, (p,)
                num_active_samples = samplewise_indicator.sum()  # number of samples supervised with NL

                if num_active_samples == 0:
                    negative_loss_mean = torch.tensor(0.0, device=rank, dtype=torch.float32)
                else:
                    negative_loss_mean = negative_loss.sum() / num_active_samples
                
                if torch.isnan(negative_loss_mean).any(): breakpoint()
                return negative_loss_mean

            for partial_loss_name in set(train_dataset.partial_loss):
                if partial_loss_name == 'negative':
                    partial_criteria.append(_negative_loss)
                else:
                    raise ValueError(f'Unsupported partial loss: {partial_loss_name}')
            partial_criterion = (
                lambda partial_probs, partial_labels: 
                sum([
                    cur_criterion(partial_probs, partial_labels)
                    for cur_criterion in partial_criteria
                ])
            )
        else:
            partial_criterion = (
                lambda partial_probs, partial_labels: 
                torch.tensor(0.0, device=rank, dtype=torch.float32)
            )

    for epoch in range(1, n_epochs + 1):
        # run.py is executed several times when loader is iterated (both in train and val)
        # This was data loader workers!

        # ... train ... #
        net.train()
        if train_sampler is not None: 
            train_sampler.set_epoch(epoch)

        n_corrects = torch.tensor(0.0, device=rank)
        losses = []
        full_losses = []
        partial_losses = []

        train_iterator = tqdm(
            enumerate(loader_tr),
            total=len(loader_tr),
            desc=f'Train... {log_prefix}epoch{epoch}',
            leave=False,
            disable=not rank == 0
        )
        
        before = time.time()
        if hasattr(train_dataset, 'partial_labels'): 
            for _, (x, y, idxs) in train_iterator: 
                x = x.to(rank)
                optimizer.zero_grad()

                loss = None
                out, _ = net(x)

                all_partial_labels = torch.from_numpy(train_dataset.partial_labels).long()[idxs].to(rank)  # (N, C)
                row_sums = torch.sum(all_partial_labels, dim=1)  # (N,)
                assert (row_sums > 0).all(), 'Samples with no labels exist'
                assert not all_partial_labels.all(), 'All samples are fully ambiguous'

                _full_label_masks = row_sums == 1

                full_labels = all_partial_labels[_full_label_masks]  # (f, C)
                if full_labels.numel() == 0:  # no full labels
                    full_loss = torch.tensor(0.0, device=rank, dtype=torch.float32)
                else:
                    full_outs = out[_full_label_masks]  # (f, C)
                    full_loss = F.cross_entropy(full_outs, torch.argmax(full_labels, dim=1).to(rank))

                partial_labels = all_partial_labels[~_full_label_masks]  # (p, C)
                if partial_labels.numel() == 0:  # no partial labels
                    partial_loss = torch.tensor(0.0, device=rank, dtype=torch.float32)
                else:
                    partial_outs = out[~_full_label_masks]  # (p, C)
                    partial_probs = F.softmax(partial_outs, dim=1)  # (p, C)
                    partial_loss = partial_criterion(partial_probs, partial_labels)
                
                loss = full_loss + train_dataset.lambda_negative * partial_loss
                
                if torch.isnan(loss).any():
                    breakpoint()  # for debugging

                if not loss.requires_grad:
                    # This is mainly because of the thresholding
                    # of negative loss
                    print(' *** no supervision available *** ')
                else:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()

                n_corrects += (torch.argmax(out, 1).detach().cpu() == y.detach().cpu()).sum().float()
                losses.append(loss.item())
                full_losses.append(full_loss.item())
                partial_losses.append(partial_loss.item())

                if wandb_run and rank == 0:
                    wandb_run.log({
                        log_prefix + 'loss': loss.item()
                    })
        else:
            for batch_idx, (x, y, idxs) in train_iterator: 
                x, y = x.to(rank), y.to(rank)
                optimizer.zero_grad()

                loss = None
                if en_mixup and (batch_idx + 1) % 3 == 0: 
                    x, y_a, y_b, lam = mixup(x, y, alpha, rank, en_mixup)
                    out, _ = net(x)
                    loss = mixup_loss(criterion, out, y_a, y_b, lam)
                else: 
                    out, e = net(x)
                    loss = criterion(out, y)
                loss.backward()

                n_corrects += (torch.argmax(out, 1) == y).sum().float()
                losses.append(loss.item())

                if wandb_run and rank == 0:
                    wandb_run.log({
                        log_prefix + 'loss': loss.item()
                    })

                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

        losses = torch.tensor(sum(losses), dtype=torch.float32, device=rank)
        full_losses = torch.tensor(sum(full_losses), dtype=torch.float32, device=rank)
        partial_losses = torch.tensor(sum(partial_losses), dtype=torch.float32, device=rank)
        n_batches = torch.tensor(len(loader_tr), dtype=torch.int64, device=rank) # number of batches in this rank

        if ddp:
            dist.all_reduce(n_corrects, op=dist.ReduceOp.SUM)
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            dist.all_reduce(full_losses, op=dist.ReduceOp.SUM)
            dist.all_reduce(partial_losses, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches, op=dist.ReduceOp.SUM)
        after = time.time()

        train_acc = n_corrects.item() / len(loader_tr.dataset)
        train_loss_avg = losses.item() / n_batches.item()
        train_full_loss_avg = full_losses.item() / n_batches.item()
        train_partial_loss_avg = partial_losses.item() / n_batches.item()

        if rank == 0: 
            output_data = [
                f'Wall-clock Time: {after - before:.2f}s',
                f'train_acc: {train_acc:.4f}',
                f'train_loss_avg: {train_loss_avg:.4f}',
                f'train_full_loss_avg: {train_full_loss_avg:.4f}',
                f'train_partial_loss_avg: {train_partial_loss_avg:.4f}'
            ]
            print(f'Train... {log_prefix}epoch{epoch} ** ' + ' ** '.join(output_data))

        scheduler.step()

        # ... validate ... #
        n_corrects = torch.tensor(0.0, device=rank)
        n_corrects_top5 = torch.tensor(0.0, device=rank)
        net.eval()
        test_iterator = tqdm(
            loader_te,
            total=len(loader_te),
            desc=f'Val... {log_prefix}epoch{epoch}',
            leave=False,
            disable=not rank == 0
        )

        before = time.time()
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
        after = time.time()
        
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
                log_prefix + 'train_full_loss_avg': train_full_loss_avg,
                log_prefix + 'train_partial_loss_avg': train_partial_loss_avg,
                log_prefix + 'val_acc': val_acc, 
                log_prefix + 'val_acc_top5': val_acc_top5
            })

    if rank == 0:
        cpu_best_model_dict = {k: v.cpu() for k, v in best_model.items()}
        fpath = _save_ckpt(cpu_best_model_dict)
        return_dict['result'] = (best_acc, best_top5_acc, fpath)

    del net, optimizer, scheduler, loader_tr, loader_te
    gc.collect()
    torch.cuda.empty_cache()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()

def _out_predict(
    rank,
    world_size,
    port,
    seed,
    ddp,
    net_constructor: Callable,
    net_ckpt_path: str,
    sync_bn,
    dataset,
    batch_size,
    num_workers,
    return_dict
): 
    net = net_constructor()
    net.load_state_dict(_load_ckpt(net_ckpt_path, remove=False))
    net = net.to(rank)

    if ddp: 
        init_for_distributed(rank, world_size, port, seed)
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
    
    del net
    gc.collect()
    
    if ddp:
        dist.barrier()
        dist.destroy_process_group()

class Strategy:
    def __init__(self, train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args):
        self.train_dataset = train_dataset
        self.train_raw_dataset = train_raw_dataset
        self.test_dataset = test_dataset
        self.idxs_lb = idxs_lb
        self.net_constructor = args['net_constructor']
        self.net = net
        self.args = args
        
        self.device = args['device']
        self.n_pool = len(train_dataset)

        self.seed = args['seed']

        self.partial_loss: list = args['partial_loss']
        self.lambda_negative = args['lambda_negative']

        self.wandb_run = None
        self.log_prefix = ''

        self.alpha = args['alpha']
        self.en_mixup = self.alpha > 0

        self.optimizer = args['optimizer']
        self.scheduler = args['scheduler']

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
        
        # NOTE: net will be initialized in the _out_train function
        # self.net = self.args['net_constructor']()
        # self.net = self.net.to(self.device)

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

        # FIXME: this is not so OOP
        if hasattr(self, 'partial_labels'):
            self.train_dataset.partial_labels = self.partial_labels
            self.train_dataset.partial_loss = self.partial_loss

            self.train_dataset.lambda_negative = self.lambda_negative

        if self.ddp:
            mp.spawn(_out_train, args=(
                self.world_size,
                self.port,
                self.seed,
                self.ddp,
                self.net_constructor,
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
                self.args['steps'],
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
                self.net_constructor,
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
                self.args['steps'],
                self.args['gamma'],
                self.en_mixup,
                self.alpha,
                self.log_prefix,
                self.wandb_run,
                return_dict
            )
        
        if 'result' not in return_dict: 
            raise ValueError('result not in return_dict')
        
        best_acc, best_top5_acc, ckpt_path = return_dict['result']
        cpu_net_dict = _load_ckpt(ckpt_path)

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

        net_ckpt_path = _save_ckpt(self.net.state_dict())

        if self.ddp:
            manager = mp.Manager()
            return_dict = manager.dict()
            mp.spawn(_out_predict, args=(
                self.world_size,
                self.port,
                self.seed,
                self.ddp,
                self.net_constructor,
                net_ckpt_path,
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
                self.net_constructor,
                net_ckpt_path,
                self.sync_bn,
                dataset,
                self.args['loader_te_args']['batch_size'],
                self.args['loader_te_args']['num_workers'],
                return_dict
            )

        _remove_ckpt(net_ckpt_path)
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
