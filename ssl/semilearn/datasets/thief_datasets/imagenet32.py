# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import os
import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
import math
import numpy as np
import pickle
from tqdm import tqdm

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]
img_size = 224

imagenet_mean = (0.4843, 0.4830, 0.4802)
imagenet_std = (0.1329, 0.1430, 0.1511)
IdentityTransform = transforms.Lambda(lambda x: x)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_imagenet32(args, alg, target_model, num_labels, num_classes, 
                    dataset_root, labeled_idxs=None, val_idxs=None):
    num_labels = num_labels // num_classes
    print('dataset root', dataset_root)

    img_size = args.img_size
    crop_ratio = args.crop_ratio
    crop_size = args.img_size

    if args.victim_dataset == 'mnist':
        weak_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
            ])

        strong_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            # transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])

        val_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])

        strong_transform2 = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),     
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),        
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])                 
    else:
        crop_size = args.img_size

        weak_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        strong_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
        ])

        strong_transform2 = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),        
            transforms.ToTensor(),
        ])                    
    

    base_dataset = ImageNet32Base(dataset_root, transform=val_transform)
    
    # Note: All labeled samples without their labels are also included in the unlabeled data
    if labeled_idxs is not None:
        labeled_set = labeled_idxs
        val_set = val_idxs
        unlabeled_set = np.array(range(len(base_dataset)))
    else:
        labeled_set, val_set, unlabeled_set = x_u_split(
            args, len(base_dataset))

    thief_data_lb = ImageNet32(alg, dataset_root, labeled_set, target_model,
                                  weak_transform=weak_transform,
                                  val_transform=val_transform)
    thief_data_val = ImageNet32(alg, dataset_root, val_set, target_model,
                                  weak_transform=val_transform, 
                                  val_transform=val_transform)
    thief_data_ulb = ImageNet32(alg, dataset_root, unlabeled_set, target_model, 
                                   is_ulb=True,
                                   weak_transform=weak_transform, 
                                   strong_transform=strong_transform, 
                                   val_transform=val_transform, 
                                   strong_transform2=strong_transform2)
    
    if alg == 'selfkdcontrastive':
        weak_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            ])
        thief_data_ulb = ImageNet32(alg, dataset_root, unlabeled_set, target_model, 
                                   is_ulb=True,
                                   weak_transform=TwoCropTransform(weak_transform),
                                   strong_transform=strong_transform, 
                                   val_transform=val_transform, 
                                   strong_transform2=strong_transform2)
    
    return thief_data_lb, thief_data_ulb, thief_data_val


def x_u_split(args, len_data):
    
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    # unlabeled_idx = np.array(range(len_data))

    # do not use the entire unlabeled set, use only SUBSET number of samples
    indices = list(range(len_data))
    random.shuffle(indices)
    indices = indices[:args.subset]
    print(len(indices))
    unlabeled_idx = indices

    # labeled data: random selection
    idx = np.arange(len_data)
    labeled_idx = np.random.choice(idx, args.num_labels, False)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labels

    num_train = args.num_labels*9//10
    train_idx = labeled_idx[:num_train]
    val_idx = labeled_idx[num_train:]

    if args.expand_labels or args.num_labels < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labels)
        train_idx = np.hstack([train_idx for _ in range(num_expand_x)])
    np.random.shuffle(train_idx)
    
    return train_idx, val_idx, unlabeled_idx


class ImageNet32(BasicDataset):
    
    def __init__(self, alg, dataset_root, indexs, 
                 victim_model, is_ulb=False, 
                 weak_transform = None, strong_transform = None, 
                 val_transform=None, 
                 strong_transform2=None):
        
        self.alg = alg
        self.is_ulb = is_ulb
        self.transform = weak_transform
        self.strong_transform = strong_transform
        self.strong_transform2 = strong_transform2

        # Use no random transforms while querying the date from the victim
        # if victim_dataset == 'mnist':
        #     base_transform = transforms.Compose([transforms.ToTensor(), 
        #                                 transforms.Normalize(imagenet_mean, imagenet_std)])
        # else:
        #     base_transform = transforms.Compose([transforms.ToTensor()])
        base_transform = val_transform
        self.base_dataset = ImageNet32Base(dataset_root, transform=base_transform) 
            
        subset_ds = Subset(self.base_dataset, indexs)
        subset_loader = DataLoader(subset_ds, batch_size=1000, num_workers=2, shuffle=False, drop_last=False, pin_memory=False)

        self.samples = []
        if self.is_ulb is not True:
            victim_model.eval()        
            with torch.no_grad():
                for d, l0, ind0 in (subset_loader):
                    d = d.cuda()
                    l = victim_model(d).argmax(axis=1, keepdim=False)
                    l = l.detach().cpu().tolist()
                    for ii, jj in enumerate(ind0):
                        self.samples.append((self.base_dataset.samples[jj][0], l[ii]))
        else:
            self.samples = [self.base_dataset.samples[e] for e in indexs]
                    
        self.data = []
        for i in range(len(self.samples)):
            self.data.append(self.samples[i][0])        
        
    def __sample__(self, index):
        img, target = self.samples[index]
        img = Image.fromarray((img.transpose((1, 2, 0))))
        return img, target

    def __getitem__(self, index):
        return BasicDataset.__getitem__(self, index)

    def __len__(self):
        return len(self.samples)


class ImageNet32Base(Dataset):
    def __init__(self, data_root, transform=None, val=False, num_shards=10):
        super().__init__()
        if val:
            filenames = [os.path.join(data_root, 'val_data')]
        else:    
            filenames = [('train_data_batch_%d' % (i+1)) for i in range(num_shards)]
            filenames = [os.path.join(data_root, f) for f in filenames]
        
        self.samples = []
        for filename in (filenames):
            if os.path.isfile(filename):
                res = unpickle(filename)
                Xs = res['data'].reshape((res['data'].shape[0],3,32,32))
                ys = res['labels']
                for i in range(len(ys)):
                    self.samples += [(Xs[i], ys[i]-1)]
        
        IdentityTransform = transforms.Lambda(lambda x: x)  
        if transform is not None:
            self.transform = transform
        else:
            self.transform = IdentityTransform

    def __getitem__(self, index):
        X, y = self.samples[index]
        # X = torch.tensor(X).float()
        X = Image.fromarray((X.transpose((1, 2, 0))).astype('uint8'))
        
        if self.transform is not None:
            X = self.transform(X)

        return X, y, index

    def __len__(self):
        return len(self.samples)








