import os

from torch.utils.data import Dataset

import h5py
import numpy as np

class SynapseDataset(Dataset):
    def __init__(self, args, dataset_dir, mode='train', transform=None, target_transform=None) -> None:
        super(SynapseDataset, self).__init__()

        self.transform = transform
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.sample_list = open(os.path.join(dataset_dir, '{}.txt'.format(mode))).readlines()
        self.num_classes = args.num_classes

        print("Dataset Length : {}".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.dataset_dir, 'train', slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            file_path = os.path.join(self.dataset_dir, 'test', vol_name + '.npy.h5')
            data = h5py.File(file_path, 'r')
            image, label = data['image'][:], data['label'][:]

        if self.num_classes == 9:
            label[label == 5] = 0
            label[label == 9] = 0
            label[label == 10] = 0
            label[label == 12] = 0
            label[label == 13] = 0
            label[label == 11] = 5

        data = {'image': image, 'target': label}
        if self.transform:
            data = self.transform(data)

        data['case_name'] = self.sample_list[idx].strip('\n')

        return data