import os

from torchvision.datasets import VisionDataset, CIFAR100
from PIL import Image

import numpy as np

class CIFAR100LT(CIFAR100): 
    def __init__(self, root, train: bool, transform=None, download=False, indices_path: str=None):
        super(CIFAR100LT, self).__init__(root, train=train, transform=transform, download=download)

        if train and indices_path is not None: 
            print(f' *** Loading an indices file at {indices_path} *** ')
            if not os.path.exists(indices_path):
                raise ValueError(f'No such file: {indices_path}')
            indices: np.ndarray = np.load(indices_path)
            indices = np.sort(indices)

            self.data = np.array(self.data)[indices]
            self.targets = np.array(self.targets)[indices]
