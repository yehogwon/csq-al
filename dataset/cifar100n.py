import os

from torchvision.datasets import VisionDataset, CIFAR100
from PIL import Image

import numpy as np

class CIFAR100N(CIFAR100): 
    def __init__(self, root, train: bool, transform=None, download=False, noisy_label_path: str=None):
        super(CIFAR100N, self).__init__(root, train=train, transform=transform, download=download)

        if train and noisy_label_path is not None: 
            print(f' *** Loading a noisy label file at {noisy_label_path} *** ')
            if not os.path.exists(noisy_label_path):
                raise ValueError(f'No such file: {noisy_label_path}')
            self.labels: list = np.load(noisy_label_path)
        else: 
            self.labels: list = self.targets
        
        assert len(self.labels) == len(self.targets)
        
    def __getitem__(self, idx):
        image, _ = super(CIFAR100N, self).__getitem__(idx)
        label = self.labels[idx]
        return image, label
