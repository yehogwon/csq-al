if __name__ == '__main__':
    print('run.py -- starting run.py', flush=True)

    import os
    import random
    import numpy as np
    import gc
    import argparse
    import torch
    from tqdm import tqdm

    from config import configs
    from dataset.common import get_dataset
    import model.vgg as vgg
    import model.resnet as resnet
    import model.efficientnet as efficientnet
    import model.mobilenet as mobilenet
    import model.wideresnet as wideresnet
    import model.preact_resnet as preact_resnet
    import model.preact_wideresnet as preact_wideresnet
    import model.csvm as csvm

    from al.strategy import set_seed
    from al import *

    import wandb

    PROJECT_NAME = 'topk-al-icml25'



    def main(cmd_args: argparse.Namespace):
        set_seed(cmd_args.seed)
        
        dataset_name = cmd_args.dataset
        initial_budget = cmd_args.initial_budget
        budget = cmd_args.budget

        if 'probcover' in cmd_args.strategy and dataset_name not in ['CIFAR10', 'CIFAR100']:
            raise ValueError('ProbCover is only supported for CIFAR10 and CIFAR100')

        configs['MNIST']['transformTest'] = configs['MNIST']['transform']
        configs['FashionMNIST']['transformTest'] = configs['FashionMNIST']['transform']
        configs['SVHN']['transformTest'] = configs['SVHN']['transform']

        args = configs[cmd_args.dataset]

        args['cmd_args'] = cmd_args
        args['budget'] = budget
        args['seed'] = cmd_args.seed

        args['modelType'] = cmd_args.model
        args['device'] = 'cuda' if cmd_args.cuda and torch.cuda.is_available() else 'cpu'
        args['wandb'] = cmd_args.wandb
        args['deterministic'] = cmd_args.deterministic
        args['rho'] = cmd_args.rho
        args['saal_batch_size'] = cmd_args.saal_batch_size
        args['alpha'] = cmd_args.alpha
        args['d'] = cmd_args.d
        args['diversity'] = cmd_args.diversity

        if args['saal_batch_size'] == 0: 
            args['saal_batch_size'] = args['loader_te_args']['batch_size']

        if args['diversity']: 
            raise NotImplementedError('SAAL with k-means++ initialization is not working')

        if cmd_args.batch_size > 0: 
            args['loader_tr_args']['batch_size'] = cmd_args.batch_size
            args['loader_te_args']['batch_size'] = cmd_args.batch_size
        
        if cmd_args.num_workers > 0:
            args['loader_tr_args']['num_workers'] = cmd_args.num_workers
            args['loader_te_args']['num_workers'] = cmd_args.num_workers

        if cmd_args.lr > 0: 
            args['lr'] = cmd_args.lr
        
        if 'milestones' not in args: 
            args['milestones'] = []
        if len(cmd_args.milestones) > 0:
            args['milestones'] = cmd_args.milestones
        
        if 'gamma' not in args: 
            args['gamma'] = 1
        if cmd_args.gamma > 0:
            args['gamma'] = cmd_args.gamma
        
        if 'weight_decay' not in args: 
            args['weight_decay'] = 0
        if cmd_args.weight_decay > 0:
            args['weight_decay'] = cmd_args.weight_decay

        if hasattr(cmd_args, 'k'):
            args['k'] = cmd_args.k
            if args['k'] >= 1: 
                args['k'] = int(args['k'])
        if cmd_args.n_epochs > 0:
            args['n_epochs'] = cmd_args.n_epochs
        
        args['calibration_set_size'] = cmd_args.calibration_set_size
        
        args['optimizer'] = cmd_args.optimizer
        args['scheduler'] = cmd_args.scheduler

        args['sync_bn'] = cmd_args.sync_bn
        args['port'] = cmd_args.port

        args['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
        args['slurm'] = args['slurm_job_id'] is not None

        args['warmup_epochs'] = cmd_args.warmup_epochs
        args['warmup_lr'] = cmd_args.warmup_lr
        args['push_warmup'] = cmd_args.push_warmup

        args['noisy_label_path'] = cmd_args.noisy_label_path
        args['indices_path'] = cmd_args.indices_path

        args['features_path'] = cmd_args.features_path
        args['delta'] = cmd_args.delta

        args['verbose'] = cmd_args.verbose

        args['cq_calib'] = cmd_args.cq_calib
        
        print(args, flush=True)

        train_dataset, train_raw_dataset, test_dataset = get_dataset(
            dataset_name, 
            train_transform=args['transform'], 
            test_transform=args['transformTest'],
            download=cmd_args.download, 
            args=args
        )

        # start experiment
        n_pool = len(train_dataset)
        n_test = len(test_dataset)

        # generate initial labeled pool
        data = {}
        if cmd_args.start_round > 1:
            if not cmd_args.load_idxs_lb:
                raise ValueError('load_idxs_lb should be specified for starting from a specific round')
            if not os.path.exists(cmd_args.load_idxs_lb):
                raise FileNotFoundError(f'{cmd_args.load_idxs_lb} not found for loading idxs_lb')
            
            # ** idxs_lb is stored after query, right before training. So not necessary to query again. **
            loaded_data = np.load(cmd_args.load_idxs_lb)
            idxs_lb = loaded_data['idxs_lb']
            data['cur_k'] = loaded_data['cur_k']
            if 'epsilon' in loaded_data: 
                data['epsilon'] = float(loaded_data['epsilon'].item())
            if 'adaptive_epsilon' in loaded_data:
                data['adaptive_epsilon'] = bool(loaded_data['adaptive_epsilon'].item())
        else: 
            idxs_lb = np.zeros(n_pool, dtype=bool)

        # load specified network
        if cmd_args.model == 'resnet18':
            net = resnet.ResNet18(n_classes=args['nClasses'])
        elif cmd_args.model == 'resnet34':
            net = resnet.ResNet34(n_classes=args['nClasses'])
        elif cmd_args.model == 'resnet50':
            net = resnet.ResNet50(n_classes=args['nClasses'])
        elif cmd_args.model == 'resnet101':
            net = resnet.ResNet101(n_classes=args['nClasses'])
        elif cmd_args.model == 'efficientnet': 
            net = efficientnet.EfficientNetV2s(n_classes=args['nClasses'])
        elif cmd_args.model == 'mobilenet': 
            net = mobilenet.MobileNetV3s(n_classes=args['nClasses'])
        elif cmd_args.model == 'wrn-28-5': # WRN-28-5
            net = wideresnet.wideresnet(depth=28, widen_factor=5, n_classes=args['nClasses'])
        elif cmd_args.model == 'wrn-36-1': # WRN-36-1
            net = wideresnet.wideresnet(depth=36, widen_factor=1, n_classes=args['nClasses'])
        elif cmd_args.model == 'wrn-36-2': # WRN-36-2
            net = wideresnet.wideresnet(depth=36, widen_factor=2, n_classes=args['nClasses'])
        elif cmd_args.model == 'wrn-36-5': # WRN-36-5
            net = wideresnet.wideresnet(depth=36, widen_factor=5, n_classes=args['nClasses'])
        elif cmd_args.model == 'vgg':
            net = vgg.VGG('VGG16', n_classes=args['nClasses'])
        elif cmd_args.model == 'preactresnet18':
            net = preact_resnet.PreActResNet18(n_classes=args['nClasses'])
        elif cmd_args.model == 'preactwideresnet18':
            net = preact_wideresnet.PreActWideResNet18(widen_factor=3, num_classes=args['nClasses'])
        elif cmd_args.model == 'svm': 
            if cmd_args.dataset != 'R52': 
                raise ValueError('SVM is only supported for R52 dataset')
            net = csvm.CSVM(n_classes=args['nClasses'], seed=cmd_args.seed, verbose=cmd_args.verbose)
        else: 
            raise ValueError('Choose a valid model: resnet18, resnet34, resnet50, resnet101, efficientnet, mobilenet, wrn-28-5, wrn-36-2, wrn-36-5, vgg, preactresnet18, preactwideresnet18, svm')

        net = net.to(args['device'])

        # set up the specified sampler
        if cmd_args.strategy == 'rand': # random sampling
            strategy = RandomSampling(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'entropy': # entropy-based sampling
            strategy = EntropySampling(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'badge': # batch active learning by diverse gradient embeddings
            strategy = BadgeSampling(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'coreset': # coreset sampling
            strategy = CoreSet(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'probcover': 
            strategy = ProbCover(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'typiclust':
            strategy = TypiClust(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'saal': 
            strategy = SAAL(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        ################################################################################################
        elif cmd_args.strategy == 'dtopk_random':
            strategy = DynamicTopKStrategyRandom(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_entropy':
            strategy = DynamicTopKStrategyEntropy(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_coreset':
            strategy = DynamicTopKStrategyCoreset(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_badge':
            strategy = DynamicTopKStrategyBadge(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_probcover':
            strategy = DynamicTopKStrategyProbCover(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_saal':
            strategy = DynamicTopKStrategySAAL(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        ################################################################################################
        elif cmd_args.strategy == 'dtopk_random_conf':
            strategy = DynamicTopKStrategyRandomConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_entropy_conf':
            strategy = DynamicTopKStrategyEntropyConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_inv_entropy_conf':
            strategy = DynamicTopKStrategyInvEntropyConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_cost_conf':
            strategy = DynamicTopKStrategyCostConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_coreset_conf':
            strategy = DynamicTopKStrategyCoresetConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_badge_conf':
            strategy = DynamicTopKStrategyBadgeConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_probcover_conf':
            strategy = DynamicTopKStrategyProbCoverConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_saal_conf':
            strategy = DynamicTopKStrategySAALConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_hybrid_entropy_conf':
            strategy = DynamicTopKStrategyHybridEntropyConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_hybrid_badge_conf':
            strategy = DynamicTopKStrategyHybridBadgeConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_hybrid_probcover_conf':
            strategy = DynamicTopKStrategyHybridProbCoverConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_hybrid_saal_conf':
            strategy = DynamicTopKStrategyHybridSAALConf(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        ################################################################################################
        elif cmd_args.strategy == 'ubdtopk_random':
            strategy = UBDynamicTopKStrategyRandom(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'ubdtopk_entropy':
            strategy = UBDynamicTopKStrategyEntropy(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'ubdtopk_badge':
            strategy = UBDynamicTopKStrategyBadge(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_entropy_conf_adap_log':
            strategy = DynamicTopKStrategyEntropyConfAdapLog(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        ################################################################################################
        elif cmd_args.strategy == 'dtopk_entropy_wall':
            strategy = DynamicTopKStrategyEntropyWall(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_badge_wall':
            strategy = DynamicTopKStrategyBadgeWall(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_random_conf_wall':
            strategy = DynamicTopKStrategyRandomConfWall(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_hybrid_entropy_conf_adap_wall':
            strategy = DynamicTopKStrategyHybridEntropyConfAdapWall(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        ########################################### Rebuttal ###########################################
        elif cmd_args.strategy == 'dtopk_random_drop': 
            strategy = DynamicTopKStrategyRandomDrop(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_entropy_drop':
            strategy = DynamicTopKStrategyEntropyDrop(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_inv_entropy_drop':
            strategy = DynamicTopKStrategyInvEntropyDrop(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        elif cmd_args.strategy == 'dtopk_badge_drop':
            strategy = DynamicTopKStrategyBadgeDrop(train_dataset, train_raw_dataset, test_dataset, idxs_lb, net, args)
        else: 
            raise ValueError('Invalid strategy name')

        if 'epsilon' in data:
            strategy.epsilon = data['epsilon']
        if 'adaptive_epsilon' in data:
            strategy.adaptive_epsilon = data['adaptive_epsilon']
        
        if cmd_args.load_net: 
            if not os.path.exists(cmd_args.load_net):
                raise FileNotFoundError(f'{cmd_args.load_net} not found for loading net')
            net.load_state_dict(torch.load(cmd_args.load_net))
            print(f'net loaded from {cmd_args.load_net}', flush=True)

        if cmd_args.wandb:
            project_name_suffix = f'-{cmd_args.project_suffix}' if len(cmd_args.project_suffix) > 0 else ''
            exp_name_prefix = f'{cmd_args.exp_note}-' if len(cmd_args.exp_note) > 0 else ''
            if cmd_args.start_round > 1:
                exp_name_prefix = 'cont-' + exp_name_prefix
            full_exp_name =f'{exp_name_prefix}{cmd_args.strategy}-{cmd_args.dataset}-{cmd_args.model}-B{cmd_args.budget}'
            logger_id = None if not args['slurm'] else f'{args["slurm_job_id"]}-{full_exp_name}'
            run = wandb.init(
                project=f'{PROJECT_NAME}{project_name_suffix}',
                id=logger_id,
                name=full_exp_name,
                config={
                    'args': args,
                    'cmd_args': vars(cmd_args)
                }
            )
            strategy.wandb_run = run

        # print info
        print(dataset_name, flush=True)
        print(type(strategy).__name__, flush=True)

        acc = np.zeros(cmd_args.n_rounds + 1)
        top5_acc = np.zeros(cmd_args.n_rounds + 1)
        for round in range(cmd_args.start_round, cmd_args.n_rounds + 1):
            log_prefix = f'round{round}/'
            strategy.set_log_prefix(log_prefix)

            if args['device'] == 'cuda': 
                torch.cuda.empty_cache()
            gc.collect()

            if cmd_args.start_round > 1 and round == cmd_args.start_round:
                # it has already been sampled, but yet to be trained
                pass
            else: 
                if round == 1:
                    idxs_tmp = np.arange(n_pool)
                    np.random.shuffle(idxs_tmp)
                    q_idxs = idxs_tmp[:initial_budget]
                else: 
                    # query
                    output = strategy.query(budget)
                    q_idxs = output
            
                # update labeled pool
                strategy.update(q_idxs)
                print(f'round{round}\tlabeled: {sum(strategy.idxs_lb)}\tunlabeled: {sum(~strategy.idxs_lb)}', flush=True)

                # save idxs_lb
                if len(cmd_args.save_idxs_lb) > 0:
                    if not os.path.exists(cmd_args.save_idxs_lb):
                        os.makedirs(cmd_args.save_idxs_lb)
                    idxs_lb_store_path = os.path.join(cmd_args.save_idxs_lb, f'round{round}.npz')
                    if isinstance(strategy, DynamicTopKStrategyConfBase):
                        np.savez(idxs_lb_store_path, idxs_lb=strategy.idxs_lb, cur_k=strategy.cur_k, epsilon=strategy.epsilon, adaptive_epsilon=strategy.adaptive_epsilon)
                    else: 
                        np.savez(idxs_lb_store_path, idxs_lb=strategy.idxs_lb, cur_k=strategy.cur_k)
                    print(f'AL properties saved at {idxs_lb_store_path}', flush=True)

            return_train: dict = strategy.train()

            val_acc = return_train['final_accuracy']
            val_acc_top5 = return_train['final_accuracy_top5']

            acc[round] = val_acc
            top5_acc[round] = val_acc_top5
            print(f'round{round}' + '\t' + str(sum(strategy.idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[round]), flush=True)
            
            if len(cmd_args.save_idxs_lb) > 0:
                if not os.path.exists(cmd_args.save_idxs_lb):
                    os.makedirs(cmd_args.save_idxs_lb)
                model_store_path = os.path.join(cmd_args.save_idxs_lb, f'round{round}_model.pth')
                torch.save(strategy.net.state_dict(), model_store_path)
                print(f'model saved at {model_store_path}', flush=True)

            if cmd_args.wandb: 
                if round == 1: 
                    strategy.wandb_run.define_metric('round')
                    strategy.wandb_run.define_metric('labeled_samples', step_metric='round')
                    strategy.wandb_run.define_metric('unlabeled_samples', step_metric='round')

                    for key in return_train.keys():
                        strategy.wandb_run.define_metric(key, step_metric='round')

                    if 'topk' in cmd_args.strategy: 
                        strategy.wandb_run.define_metric('certain_ratio', step_metric='round')
                        strategy.wandb_run.define_metric('ambiguous_ratio', step_metric='round')
                strategy.wandb_run.log({
                    'round': round,
                    'labeled_samples': sum(strategy.idxs_lb),
                    'unlabeled_samples': sum(~strategy.idxs_lb)
                })
                    
                strategy.wandb_run.define_metric(log_prefix + 'loss')
                strategy.wandb_run.define_metric(log_prefix + 'epoch')
                strategy.wandb_run.define_metric(log_prefix + 'train_acc', step_metric=log_prefix + 'epoch')
                strategy.wandb_run.define_metric(log_prefix + 'train_loss_avg', step_metric=log_prefix + 'epoch')
                strategy.wandb_run.define_metric(log_prefix + 'val_acc', step_metric=log_prefix + 'epoch')
                strategy.wandb_run.define_metric(log_prefix + 'val_acc_top5', step_metric=log_prefix + 'epoch')

                strategy.wandb_run.log(return_train)
                
            if strategy.stop_condition(): break
            
        if cmd_args.wandb: 
            strategy.wandb_run.finish()

    ########### START FROM HERE ###########
    strategies = [
        'rand',
        'entropy',
        'badge',
        'coreset',
        'saal',
        'probcover',
        'typiclust'
    ] + [
        'dtopk_random',
        'dtopk_entropy',
        'dtopk_coreset',
        'dtopk_badge',
        'dtopk_probcover',
        'dtopk_saal',
        'dtopk_random_conf',
        'dtopk_entropy_conf',
        'dtopk_inv_entropy_conf',
        'dtopk_coreset_conf',
        'dtopk_badge_conf',
        'dtopk_probcover_conf',
        'dtopk_saal_conf',
        'dtopk_cost_conf',
        'dtopk_hybrid_entropy_conf',
        'dtopk_hybrid_badge_conf',
        'dtopk_hybrid_probcover_conf',
        'dtopk_hybrid_saal_conf',
    ] + [
        'ubdtopk_random',
        'ubdtopk_entropy',
        'ubdtopk_badge',
    ] + [
        'dtopk_entropy_conf_adap_log'
    ] + [
        'dtopk_entropy_wall', 
        'dtopk_badge_wall', 
        'dtopk_random_conf_wall',
        'dtopk_hybrid_entropy_conf_adap_wall'
    ] + [ # Rebuttal: CSQ-sift
        'dtopk_random_drop',
        'dtopk_entropy_drop',
        'dtopk_inv_entropy_drop',
        'dtopk_badge_drop'
    ]
    
    parser = argparse.ArgumentParser()

    ### Active Learning
    parser.add_argument('--strategy', type=str, required=True, choices=strategies, help='selection strategy for AL')
    parser.add_argument('--start_round', type=int, default=1, help='starting round')
    parser.add_argument('--n_rounds', type=int, default=500, help='number of rounds')
    parser.add_argument('--initial_budget', type=int, default=100, help='initial budget')
    parser.add_argument('--budget', help='number of points to query in a batch', type=int, default=0)

    ### Active Learning Load & Store
    parser.add_argument('--load_idxs_lb', type=str, default='', help='path to load idxs_lb')
    parser.add_argument('--load_net', type=str, default='', help='path to load net')
    parser.add_argument('--save_idxs_lb', type=str, default='', help='path to save idxs_lb')

    ### CSQ
    parser.add_argument('--k', type=float, default=3, help='k value for naive top-k query')
    parser.add_argument('--calibration_set_size', type=int, default=0, help='size of calibration set for conformal prediction. If zero, it uses full val set as calibration set. Otherwise, use this number of samples from most recently labeled ones on each round for calibration')
    parser.add_argument('--cq_calib', action='store_true', help='Query with CQ for calibration set')

    ### Sampling
    parser.add_argument('--deterministic', action='store_true', help='do not use randomness in BADGE sampling')
    parser.add_argument('--d', type=float, default=0.0, help='hyperparameter for cost-efficient sampling')
    parser.add_argument('--rho', type=float, default=0.05, help='norm restriction for SAAL')
    parser.add_argument('--saal_batch_size', type=int, default=0, help='batch size for SAAL (default: batch size of test loader)')
    parser.add_argument('--diversity', action='store_true', help='use diversity for SAAL')
    parser.add_argument('--features_path', type=str, default='', help='path to dataset features')
    parser.add_argument('--delta', type=float, default=0.1, help='delta for ProbCover')

    ### Dataset
    parser.add_argument('--dataset', type=str.upper, required=True, choices=['MNIST', 'FASHIONMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'IMAGENET32', 'IMAGENET64', 'CIFAR100N', 'CIFAR100LT', 'R52', 'CUB', 'TINYIMAGENET'], help='dataset name')
    parser.add_argument('--download', action='store_true', help='download dataset')
    parser.add_argument('--noisy_label_path', type=str, default='', help='path to noisy labels; required only when dataset is CIFAR100N')
    parser.add_argument('--indices_path', type=str, default='', help='path to indices for CIFAR100LT')

    ### Training
    parser.add_argument('--model', help='model to train', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet', 'mobilenet', 'wrn-28-5', 'wrn-36-1', 'wrn-36-2', 'wrn-36-5', 'vgg', 'preactresnet18', 'preactwideresnet18', 'svm'])
    parser.add_argument('--lr', type=float, default=0, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=0, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.0, help='alpha for mix-up beta distribution')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW'], help='specify optimizer')
    parser.add_argument('--scheduler', type=str, default='MultiStepLR', choices=['MultiStepLR', 'CosineAnnealingLR'], help='specify scheduler')
    parser.add_argument('--milestones', type=int, nargs='*', default=[], help='milestones for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0, help='gamma for learning rate scheduler')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')

    ### DDP
    parser.add_argument('--sync_bn', action='store_true', help='use sync batch norm')
    parser.add_argument('--port', type=int, default=3000, help='port number for ddp')

    ### Warmup
    parser.add_argument('--warmup_epochs', type=int, default=0, help='number of warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0, help='warmup learning rate (at the beginning of training)')
    parser.add_argument('--push_warmup', action='store_true', help='push warmup to the beginning of training (n_epochs is increased by warmup_epochs)')
    
    ### Logging
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--project_suffix', type=str, default='', help='wandb project name suffix')
    parser.add_argument('--exp_note', type=str, default='', help='wandb experiment identifier')
    parser.add_argument('--verbose', action='store_true', help='verbose; print quite a lot of information')

    ### etc.
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    cmd_args = parser.parse_args()

    print('::::: PID :::::', os.getpid(), flush=True)
    print(cmd_args, flush=True)
    main(cmd_args)
