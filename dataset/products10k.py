import os
from os.path import join as pjoin

from torchvision.datasets import VisionDataset
from PIL import Image

import pandas as pd
from glob import glob

class Products10K(VisionDataset): 
    def __init__(self, root, train: bool, transform=None):
        super(Products10K, self).__init__(root, transform=transform)
        
        self.split = 'train' if train else 'test'
        dir_path = pjoin(self.root, self.split)

        self.image_paths = glob(pjoin(dir_path, '*.jpg'))

        label_path = pjoin(self.root, f'{self.split}.csv')
        label_df = pd.read_csv(label_path)
        fname_label_map = dict(zip(label_df['name'], label_df['class']))

        assert len(self.image_paths) == len(fname_label_map), 'Mismatch in numbers of images and labels'
        
        self.targets = [int(fname_label_map[os.path.basename(fpath)]) for fpath in self.image_paths]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB')
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label
