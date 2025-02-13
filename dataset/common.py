import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets
from torch.utils.data import Dataset, Subset, TensorDataset
from PIL import Image
from torchvision import transforms

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.datasets import fetch_rcv1
from datasets import load_dataset

from . import cub
from . import tiny_imagenet
from . import imagenet32
from . import imagenet64

from . import cifar100n
from . import cifar100lt

class DatasetWrapper(Dataset): 
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index): 
        item = self.dataset[index]
        if not isinstance(item, tuple): 
            item = (item,)
        return item + (index,)
    
    def __len__(self):
        return len(self.dataset)

def get_dataset(name, train_transform=None, test_transform=None, download=False, args: dict=None):
    path = os.path.dirname(os.path.realpath(__file__))
    if name == 'MNIST':
        return get_MNIST(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'FASHIONMNIST':
        return get_FashionMNIST(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'SVHN':
        return get_SVHN(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'CIFAR10':
        return get_CIFAR10(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'CIFAR100':
        return get_CIFAR100(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'CIFAR100N':
        if args is None or 'noisy_label_path' not in args: 
            raise ValueError('noisy_label_path must be provided')
        return get_CIFAR100N(path, train_transform=train_transform, test_transform=test_transform, download=download, noisy_label_path=args['noisy_label_path'])
    elif name == 'CIFAR100LT':
        if args is None or 'indices_path' not in args: 
            raise ValueError('indices_path must be provided')
        return get_CIFAR100LT(path, train_transform=train_transform, test_transform=test_transform, download=download, indices_path=args['indices_path'])
    elif name == 'CUB': 
        return get_CUB(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'IMAGENET': 
        return get_ImageNet(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'TINYIMAGENET': 
        return get_TinyImageNet(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'IMAGENET32':
        return get_ImageNet32(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'IMAGENET64':
        return get_ImageNet64(path, train_transform=train_transform, test_transform=test_transform, download=download)
    elif name == 'R52': 
        return get_R52()

def get_MNIST(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'MNIST')
    train = datasets.MNIST(root, train=True, transform=train_transform, download=download)
    train_test = datasets.MNIST(root, train=True, transform=test_transform, download=download)
    test = datasets.MNIST(root, train=False, transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_FashionMNIST(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'FashionMNIST')
    train = datasets.FashionMNIST(root, train=True, transform=train_transform, download=download)
    train_test = datasets.FashionMNIST(root, train=True, transform=test_transform, download=download)
    test = datasets.FashionMNIST(root, train=False, transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_SVHN(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'SVHN')
    train = datasets.SVHN(root, split='train', transform=train_transform, download=download)
    train_test = datasets.SVHN(root, split='train', transform=test_transform, download=download)
    test = datasets.SVHN(root, split='test', transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_CIFAR10(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'CIFAR10')
    train = datasets.CIFAR10(root, train=True, transform=train_transform, download=download)
    train_test = datasets.CIFAR10(root, train=True, transform=test_transform, download=download)
    test = datasets.CIFAR10(root, train=False, transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_CIFAR100(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'CIFAR100')
    train = datasets.CIFAR100(root, train=True, transform=train_transform, download=download)
    train_test = datasets.CIFAR100(root, train=True, transform=test_transform, download=download)
    test = datasets.CIFAR100(root, train=False, transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_CIFAR100N(path, train_transform=None, test_transform=None, download=False, noisy_label_path: str=None):
    if noisy_label_path is None: 
        raise ValueError('noisy_label_path must be provided')
    root = os.path.join(path, 'CIFAR100')
    train = cifar100n.CIFAR100N(root, train=True, transform=train_transform, download=download, noisy_label_path=noisy_label_path)
    train_test = cifar100n.CIFAR100N(root, train=True, transform=test_transform, download=download, noisy_label_path=noisy_label_path)
    test = cifar100n.CIFAR100N(root, train=False, transform=test_transform, download=download, noisy_label_path=noisy_label_path)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_CIFAR100LT(path, train_transform=None, test_transform=None, download=False, indices_path: str=None):
    if indices_path is None: 
        raise ValueError('indices_path must be provided')
    root = os.path.join(path, 'CIFAR100')
    train = cifar100lt.CIFAR100LT(root, train=True, transform=train_transform, download=download, indices_path=indices_path)
    train_test = cifar100lt.CIFAR100LT(root, train=True, transform=test_transform, download=download, indices_path=indices_path)
    test = cifar100lt.CIFAR100LT(root, train=False, transform=test_transform, download=download, indices_path=indices_path)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_CUB(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'CUB')
    train = cub.Cub2011(root, train=True, transform=train_transform, download=download)
    train_test = cub.Cub2011(root, train=True, transform=test_transform, download=download)
    test = cub.Cub2011(root, train=False, transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_ImageNet(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'ImageNet')
    train = datasets.ImageNet(root, split='train', transform=train_transform, download=download)
    train_test = datasets.ImageNet(root, split='train', transform=test_transform, download=download)
    test = datasets.ImageNet(root, split='val', transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_TinyImageNet(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'TinyImageNet')
    train = tiny_imagenet.TinyImageNet(root, split='train', transform=train_transform, download=download)
    train_test = tiny_imagenet.TinyImageNet(root, split='train', transform=test_transform, download=download)
    test = tiny_imagenet.TinyImageNet(root, split='val', transform=test_transform, download=download)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_ImageNet32(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'ImageNet32')
    train = imagenet32.ImageNet32(root, train=True, transform=train_transform)
    train_test = imagenet32.ImageNet32(root, train=True, transform=test_transform)
    test = imagenet32.ImageNet32(root, train=False, transform=test_transform)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_ImageNet64(path, train_transform=None, test_transform=None, download=False):
    root = os.path.join(path, 'ImageNet64')
    train = imagenet64.ImageNet64(root, train=True, transform=train_transform)
    train_test = imagenet64.ImageNet64(root, train=True, transform=test_transform)
    test = imagenet64.ImageNet64(root, train=False, transform=test_transform)
    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)

def get_R52(): 
    dataset = load_dataset('dxgp/R52')
    train, _, test = dataset['train'], dataset['validation'], dataset['test']
    
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)

    vectorizer = TfidfVectorizer()

    x_train = vectorizer.fit_transform(train_df['text'])
    y_train = torch.tensor(train_df['label'], dtype=torch.int64)

    x_test = vectorizer.transform(test_df['text'])
    y_test = torch.tensor(test_df['label'], dtype=torch.int64)

    train = TensorDataset(torch.tensor(x_train.toarray(), dtype=torch.float32), y_train)
    train_test = TensorDataset(torch.tensor(x_train.toarray(), dtype=torch.float32), y_train)
    test = TensorDataset(torch.tensor(x_test.toarray(), dtype=torch.float32), y_test)

    return DatasetWrapper(train), DatasetWrapper(train_test), DatasetWrapper(test)
