import os
import random

import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

class PolypImageSegDataset(Dataset) :
    def __init__(self, args, dataset_dir, mode='train', transform=None, target_transform=None):
        super(PolypImageSegDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.image_folder = 'images'
        self.label_folder = 'masks'
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        if mode == 'train':
            self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))
        else:
            self.frame = pd.read_csv(os.path.join('/'.join(dataset_dir.split('/')[:-1]), 'PolypSegData', '{}_{}_frame.csv'.format(dataset_dir.split('/')[-1], mode)))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if self.mode == 'train':
            image_path = os.path.join(self.dataset_dir, 'TrainDataset', self.image_folder, self.frame.image_path[idx])
            target_path = os.path.join(self.dataset_dir, 'TrainDataset', self.label_folder, self.frame.mask_path[idx])
        elif self.mode == 'test':
            image_path = os.path.join(os.path.join('/'.join(self.dataset_dir.split('/')[:-1]), 'PolypSegData', 'TestDataset', 'TestDataset', self.dataset_dir.split('/')[-1], self.image_folder, self.frame.image_path[idx]))
            target_path = os.path.join(os.path.join('/'.join(self.dataset_dir.split('/')[:-1]), 'PolypSegData', 'TestDataset', 'TestDataset', self.dataset_dir.split('/')[-1], self.label_folder, self.frame.mask_path[idx]))

        image = Image.open(image_path).convert('RGB')
        target = Image.open(target_path).convert('L')

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); target = self.target_transform(target)

        target[target >= 0.5] = 1; target[target < 0.5] = 0

        data_dict = {'image': image,
                     'target': target}

        return data_dict

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)