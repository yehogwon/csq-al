from torchvision import transforms

configs = {
    'MNIST': {
        'nClasses': 10,
        'n_epochs': 100, 
        'transform': transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ]), 
        'loader_tr_args': {
            'batch_size': 64, 
            'num_workers': 8
        },
        'loader_te_args': {
            'batch_size': 1000, 
            'num_workers': 8
        }
    },
    'FashionMNIST': {
        'nClasses': 10,
        'n_epochs': 100, 
        'transform': transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'loader_tr_args': {
            'batch_size': 64, 
            'num_workers': 8
        },
        'loader_te_args':{
            'batch_size': 1000, 
            'num_workers': 8
        }
    },
    'SVHN': {
        'nClasses': 10,
        'n_epochs': 100, 
        'transform': transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ]),
        'loader_tr_args': {
            'batch_size': 64, 
            'num_workers': 8
        },
        'loader_te_args': {
            'batch_size': 1000,
            'num_workers': 8
        }
    },
    'CIFAR10': {
        'nClasses': 10,
        'n_epochs': 200, 
        'lr': 0.001, 
        'milestones': [60, 120, 160],
        'gamma': 0.2,
        'weight_decay': 5e-4,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]),
        'loader_tr_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'loader_te_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    },
    'CIFAR100': {
        # From the above (CIFAR10) and https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy
        'nClasses': 100,
        'n_epochs': 200, 
        'lr': 0.001, 
        'milestones': [60, 120, 160],
        'gamma': 0.2,
        'weight_decay': 5e-4,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        'loader_tr_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'loader_te_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    },
    'CIFAR100N': {
        # From the above (CIFAR10) and https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy
        'nClasses': 100,
        'n_epochs': 200, 
        'lr': 0.001, 
        'milestones': [60, 120, 160],
        'gamma': 0.2,
        'weight_decay': 5e-4,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        'loader_tr_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'loader_te_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    },
    'CIFAR100LT': {
        # From the above (CIFAR10) and https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy
        'nClasses': 100,
        'n_epochs': 200, 
        'lr': 0.001, 
        'milestones': [60, 120, 160],
        'gamma': 0.2,
        'weight_decay': 5e-4,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]),
        'loader_tr_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'loader_te_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    },
    'IMAGENET32': {
        'nClasses': 1000,
        'n_epochs': 200, 
        'lr': 0.01, 
        'milestones': [60, 120, 160],
        'gamma': 0.2,
        'weight_decay': 5e-4,
        'transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], (0.229, 0.224, 0.225)) # not accurate
        ]),
        'loader_tr_args': {
            'batch_size': 512, 
            'num_workers': 8
        },
        'loader_te_args': {
            'batch_size': 512, 
            'num_workers': 8
        },
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], (0.229, 0.224, 0.225)) # not accurate
        ])
    },
    'IMAGENET64': {
        'nClasses': 1000,
        'n_epochs': 30, 
        'lr': 0.002, 
        'milestones': [10, 20, 30],
        'gamma': 0.2,
        'weight_decay': 0,
        'transform': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(4/64, 4/64)),
            transforms.ToTensor(),
            transforms.Normalize([0.481, 0.458, 0.408], (0.269, 0.261, 0.276))
        ]),
        'loader_tr_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'loader_te_args': {
            'batch_size': 128, 
            'num_workers': 2
        },
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.481, 0.458, 0.408], (0.269, 0.261, 0.276))
        ])
    },
    'TINYIMAGENET': {
        'nClasses': 200,
        'n_epochs': 200, 
        'lr': 0.2, 
        'weight_decay': 0.0001,
        'transform': transforms.Compose([
            transforms.RandomResizedCrop(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], (0.229, 0.224, 0.225))
        ]), 
        'transformTest': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], (0.229, 0.224, 0.225))
        ]),
        'loader_tr_args': {
            'batch_size': 128, 
            'num_workers': 4
        },
        'loader_te_args': {
            'batch_size': 128, 
            'num_workers': 4
        }
    },
    'CUB': {
        'nClasses': 200,
        'n_epochs': 200, 
        'lr': 0.01, 
        'milestones': [60, 120, 160],
        'gamma': 0.2,
        'weight_decay': 5e-4,
        'transform': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]), 
        'transformTest': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'loader_tr_args': {
            'batch_size': 64, 
            'num_workers': 4
        },
        'loader_te_args': {
            'batch_size': 64, 
            'num_workers': 4
        }
    },
    'R52': {
        'nClasses': 52,
        'n_epochs': -1, 
        'lr': -1, 
        'milestones': [],
        'gamma': -1,
        'weight_decay': -1,
        'transform': None,
        'loader_tr_args': None,
        'loader_te_args': None,
        'transformTest': None
    }
}
