import os

from torchvision.datasets import VisionDataset
from PIL import Image

import numpy as np
import pickle

# train 1 ~ 9: 128116
# train 10: 128123
# val: 50000

class ImageNet32(VisionDataset): 
    def __init__(self, root, train: bool, transform=None):
        super(ImageNet32, self).__init__(root, transform=transform)
        
        self.train = train
        dir_path = os.path.join(self.root, 'train' if train else 'val')

        self.image_paths = []
        for dir_name in os.listdir(dir_path): 
            if 'DS_Store' in dir_name:
                continue
            for file_name in os.listdir(os.path.join(dir_path, dir_name)): 
                if 'DS_Store' in file_name:
                    continue
                self.image_paths.append(os.path.join(dir_path, dir_name, file_name))

        # raise an exception if any one of self.image_paths does not end with .png
        for image_path in self.image_paths: 
            assert image_path.endswith('.png')
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB')
        label = int(image_path.split('/')[-2])
        
        if self.transform:
            image = self.transform(image)

        return image, label
