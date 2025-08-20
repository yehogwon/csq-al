import os

from torchvision.datasets import VisionDataset
from PIL import Image

class TinyImageNet(VisionDataset): 
    def __init__(self, root, split='train', transform=None, download=False):
        root = os.path.join(root, 'tiny-imagenet-200')
        super(TinyImageNet, self).__init__(root, transform=transform)
        self.root_dir = root
        self.split = split

        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be 'train', 'val', or 'test'.")
        
        assert split != 'test', 'Test split not implemented yet.'
        assert download is False, 'Download is not supported for TinyImageNet. Please download the dataset manually.'

        wnids_file = os.path.join(root, 'wnids.txt')
        with open(wnids_file, 'r') as f:
            self.cls_ids = f.read().splitlines() # list of n*** ids
            self.cls_lookup = {cls_id: i for i, cls_id in enumerate(self.cls_ids)} # n* -> index

        if split == 'train':
            self.image_folder = os.path.join(root, 'train')
            self.image_paths = []
            self.labels = []
            for class_name in os.listdir(self.image_folder):
                class_id = self.cls_lookup[class_name]
                class_dir = os.path.join(self.image_folder, class_name)
                for image_name in os.listdir(os.path.join(class_dir, 'images')):
                    self.image_paths.append(os.path.join(class_dir, 'images', image_name))
                    self.labels.append(class_id)
        elif split == 'val':
            self.image_folder = os.path.join(root, 'val/images')
            self.image_paths = [os.path.join(self.image_folder, fname) for fname in os.listdir(self.image_folder)]
            val_annotations_path = os.path.join(root, 'val/val_annotations.txt')
            with open(val_annotations_path, 'r') as f:
                annotations = {line.split('\t')[0]: self.cls_lookup[line.split('\t')[1]] for line in f.readlines()}
            self.labels = [annotations[os.path.basename(path)] for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
