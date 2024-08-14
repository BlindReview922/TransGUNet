import os

from torch.utils.data import Dataset

import h5py
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.ndimage import zoom

class AMOS2022Dataset(Dataset):
    def __init__(self, args, dataset_dir, mode='train', transform=None, target_transform=None) -> None:
        super(AMOS2022Dataset, self).__init__()

        self.transform = transform
        self.mode = mode
        self.dataset_dir = dataset_dir
        print(dataset_dir)
        if args.test_data_type == 'AMOS2022':
            self.sample_list = glob(os.path.join(dataset_dir, "{}_npz_new".format(mode), "*.h5"))
        else:
            self.sample_list = glob(os.path.join(dataset_dir, "mri_{}_npz_new".format(mode), "*.h5"))

        self.num_classes = args.num_classes

        print("Dataset Length : {}".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        vol_name = self.sample_list[idx].strip('\n')
        data = h5py.File(vol_name, 'r')
        image, label = data['image'][:], data['label'][:]

        x, y = image.shape[1], image.shape[2]

        if self.num_classes == 9:
            label[label == 5] = 0
            label[label == 9] = 0
            label[label == 10] = 5
            label[label == 11] = 0
            label[label == 12] = 0
            label[label == 13] = 0
            label[label == 14] = 0
            label[label == 15] = 0

        # image_resize = np.zeros((image.shape[0], 512, 512))
        # label_resize = np.zeros((image.shape[0], 512, 512))

        # for slice_idx in range(image.shape[0]):
        #     image_slice, label_slice = image[slice_idx], label[slice_idx]
        #     image_slice = zoom(image_slice, (512 / x, 512 / y), order=3)
        #     label_slice = zoom(label_slice, (512 / x, 512 / y), order=0)
        #
        #     image_resize[slice_idx] = image_slice
        #     label_resize[slice_idx] = label_slice

            # if np.sum(label_slice) == 0: continue
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(image_slice, cmap='gray')
            # ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])
            # ax[1].imshow(label_slice, cmap='gray')
            # ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([])
            # plt.tight_layout()
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            # plt.show()

        data = {'image': image, 'target': label}

        if self.transform:
            data = self.transform(data)

        data['case_name'] = vol_name.split("/")[-1].split(".")[0]

        return data