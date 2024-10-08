import os
import random

import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

class STUSegmentationDataset(Dataset):
    def __init__(self, args, dataset_dir, mode, transform=None, target_transform=None):
        super(STUSegmentationDataset, self).__init__()
        self.args = args
        self.image_folder = 'image'
        self.label_folder = 'mask'
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_folder, self.frame.image_path[idx])
        label_path = os.path.join(self.dataset_dir, self.label_folder, self.frame.mask_path[idx])

        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.target_transform(label)

        label[label >= 0.5] = 1; label[label < 0.5] = 0

        data_dict = {'image': image,
                     'target': label}

        return data_dict

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)