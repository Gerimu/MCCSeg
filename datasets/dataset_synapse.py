import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image

def random_rot_flip(image, label, edge):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    edge = np.rot90(edge, k)                                # edge gao
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    edge = np.flip(edge, axis=axis).copy()
    return image, label, edge


def random_rotate(image, label, edge):                            #edge gao
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    edge = ndimage.rotate(edge, angle, order=0, reshape=False)
    return image, label, edge


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, edge = sample['image'], sample['label'], sample['edge']   #edge gao

        if random.random() > 0.5:
            image, label, edge = random_rot_flip(image, label, edge)
        elif random.random() > 0.5:
            image, label, edge = random_rotate(image, label, edge)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            edge  = zoom(edge, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        edge = torch.from_numpy(edge.astype(np.float32))

        sample = {'image': image, 'label': label.long(), 'edge': edge.long() }
        return sample


# class Synapse_dataset(Dataset):
#     def __init__(self, base_dir, edge_dir, list_dir, split, transform=None):  #edge_dir for train
#         self.transform = transform  # using transform in torch!
#         self.split = split
#         self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
#         self.data_dir = base_dir
#         self.edge_dir = edge_dir     #edge gao
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     def __getitem__(self, idx):
#         if self.split == "train":
#             slice_name = self.sample_list[idx].strip('\n')
#             data_path = os.path.join(self.data_dir, slice_name+'.npz')
#             edge_path = os.path.join(self.edge_dir, slice_name+'.png')   #edge gao
#             data = np.load(data_path)
#             I = Image.open(edge_path)
#             edge = np.array(I)
#             image, label = data['image'], data['label']
#
#         else:
#             vol_name = self.sample_list[idx].strip('\n')
#             filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
#             data = h5py.File(filepath)
#             image, label = data['image'][:], data['label'][:]
#
#         # sample = {'image': image, 'label': label}
#         sample = {'image': image, 'label': label, 'edge': edge}   # edge gao
#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.sample_list[idx].strip('\n')
#         return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


