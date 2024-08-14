import os
from time import time

import h5py
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

data_root_path = '/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/IS2D_dataset/BioMedicalDataset'
data_type = 'AMOS2022'
data_path = os.path.join(data_root_path, data_type)

splits = ['test']

for split in splits:
    if split == 'train':
        ct_path = os.path.join(data_path, 'imagesTr')
        seg_path = os.path.join(data_path, 'labelsTr')
        save_path = os.path.join(data_path, 'train_npz_new')
    else:
        ct_path = os.path.join(data_path, 'imagesVa')
        seg_path = os.path.join(data_path, 'labelsVa')
        save_path = os.path.join(data_path, 'mri_test_npz_new')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("CT Path: {}".format(ct_path))
    print("Seg Path: {}".format(seg_path))
    print("Save Path: {}".format(save_path))

    upper, lower = 275, -125

    start_time = time()

    for ct_file_name in os.listdir(ct_path):
        if not ct_file_name.endswith('.nii.gz'): continue
        case_number = int(ct_file_name.split('_')[-1].split('.')[0])
        if case_number < 500: continue
        print(ct_file_name)


        ct_file = nib.load(os.path.join(ct_path, ct_file_name))
        seg_file = nib.load(os.path.join(seg_path, ct_file_name))

        # Convert to numpy array
        ct_array = ct_file.get_fdata()
        seg_array = seg_file.get_fdata()

        ct_array = np.clip(ct_array, lower, upper)
        ct_array = (ct_array - lower) / (upper - lower)

        ct_array = np.transpose(ct_array, (2, 0, 1))
        seg_array = np.transpose(seg_array, (2, 0, 1))

        x, y = ct_array.shape[1], ct_array.shape[2]

        if ct_array.shape[1] != 512 or ct_array.shape[2] != 512:
            new_ct_array = np.zeros((ct_array.shape[0], 512, 512))
            new_seg_array = np.zeros((seg_array.shape[0], 512, 512))

            for slice_idx in range(ct_array.shape[0]):
                ct_array_slice = ct_array[slice_idx]
                seg_array_slice = seg_array[slice_idx]
                new_ct_array[slice_idx] = zoom(ct_array_slice, (512 / x, 512 / y), order=3)
                new_seg_array[slice_idx] = zoom(seg_array_slice, (512 / x, 512 / y), order=0)

            ct_array = new_ct_array
            seg_array = new_seg_array

        # Delete zero slice in ct_array and seg_array
        num_zero_slice = 0
        for slice_idx in range(ct_array.shape[0]):
            if np.sum(seg_array[slice_idx]) == 0 or np.sum(ct_array[slice_idx]) == 0:
                num_zero_slice += 1

        # print('num_zero_slice: {}'.format(num_zero_slice))
        new_ct_array = np.zeros((ct_array.shape[0] - num_zero_slice, ct_array.shape[1], ct_array.shape[2]))
        new_seg_array = np.zeros((seg_array.shape[0] - num_zero_slice, seg_array.shape[1], seg_array.shape[2]))
        current_slice_idx = 0
        for slice_idx in range(ct_array.shape[0]):
            if np.sum(seg_array[slice_idx]) == 0 or np.sum(ct_array[slice_idx]) == 0:
                continue
            new_ct_array[current_slice_idx] = ct_array[slice_idx]
            new_seg_array[current_slice_idx] = seg_array[slice_idx]
            current_slice_idx += 1

        ct_array = new_ct_array
        seg_array = new_seg_array

        ct_number = ct_file_name.split('.')[0]

        if split == 'test':
            new_ct_name = ct_number + '.npy.h5'
            hf = h5py.File(os.path.join(save_path, new_ct_name), 'w')
            hf.create_dataset('image', data=ct_array)
            hf.create_dataset('label', data=seg_array)
            hf.close()
            continue

        for slice_idx in range(ct_array.shape[0]):
            ct_array_slice = ct_array[slice_idx]
            seg_array_slice = seg_array[slice_idx]
            slice_no = "{:03d}".format(slice_idx)
            new_ct_name = ct_number + '_slice' + slice_no
            np.savez(os.path.join(save_path, new_ct_name), image=ct_array_slice, label=seg_array_slice)

        print('already use {:.3f} min'.format((time() - start_time) / 60))
        print('-----------')