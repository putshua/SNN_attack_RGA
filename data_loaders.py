import json
import os
import random
import warnings
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, ImageFolder

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST

warnings.filterwarnings('ignore')

# code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py 
# Improved Regularization of Convolutional Neural Networks with Cutout.
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def build_cifar(cutout=True, use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        # aug.append(
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root='~/datasets/',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='~/datasets/',
                              train=False, download=download, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR100(root='~/datasets/',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='~/datasets/',
                               train=False, download=download, transform=transform_test)
        norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    return train_dataset, val_dataset, norm

def build_svhn(cutout=True, download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(cutout(n_holes=1, length=16))

    transform_train = transforms.Compose(aug)
    transform_test = transforms.ToTensor()
    train_dataset = SVHN(root='~/datasets/', split='train', download=download, transform=transform_train)
    val_dataset = SVHN(root='~/datasets/', split='test', download=download, transform=transform_test)
    norm = ((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    return train_dataset, val_dataset, norm

def build_dvscifar(root):
    def t(data):
        aug = transforms.Compose([transforms.Resize(size=(48, 48)), transforms.RandomHorizontalFlip()])
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data
    
    def tt(data):
        aug = transforms.Resize(size=(48, 48))
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data

    data1 = CIFAR10DVS(root=root, data_type='frame', frames_number=10, split_by='number', transform=t)
    train_dataset, _ = torch.utils.data.random_split(data1, [9000, 1000], generator=torch.Generator().manual_seed(42))
    data2 = CIFAR10DVS(root=root, data_type='frame', frames_number=10, split_by='number', transform=tt)
    _, val_dataset = torch.utils.data.random_split(data2, [9000, 1000], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset, None

def build_dvsgesture(root):
    def t(data):
        aug = transforms.Compose([transforms.Resize(size=(64, 64)), transforms.RandomHorizontalFlip()])
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data
    
    def tt(data):
        aug = transforms.Resize(size=(64, 64))
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data

    train_dataset = DVS128Gesture(root=root, train=True, data_type='frame', frames_number=10, split_by='number', transform=t)
    val_dataset = DVS128Gesture(root=root, train=False, data_type='frame', frames_number=10, split_by='number', transform=tt)

    return train_dataset, val_dataset, None

def build_nmnist(root):
    def t(data):
        aug = transforms.RandomHorizontalFlip()
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data
    
    def tt(data):
        data = torch.from_numpy(data).float()
        return data
    train_dataset = NMNIST(root=root, train=True, data_type='frame', frames_number=10, split_by='number', transform=t)
    val_dataset = NMNIST(root=root, train=False, data_type='frame', frames_number=10, split_by='number', transform=tt)
    return train_dataset, val_dataset, None

def build_imagenet(root):
    train_root = os.path.join(root, 'ILSVRC2012_train')
    val_root = os.path.join(root, 'ILSVRC2012_val')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    norm = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return train_dataset, val_dataset, norm