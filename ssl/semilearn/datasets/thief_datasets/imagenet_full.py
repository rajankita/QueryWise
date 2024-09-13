#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp
import csv
import random
import math
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.datasets import imagenet
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
# from .augmix import AugMix
# import knockoff.config as cfg
# import natsort
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio
from copy import deepcopy
import json
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
import torch


def imageio_loader(path: str):
    with open(path, "rb") as f:
        img = imageio.imread(f)
        img = img.permute(2,1,0)
        return img
    

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_imagenet_full(args, alg, target_model, num_labels, num_classes, 
                       dataset_root, labeled_idxs=None, val_idxs=None):
    num_labels = num_labels // num_classes
    print('dataset root', dataset_root)

    img_size = args.img_size
    crop_ratio = args.crop_ratio

    weak_transform = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    strong_transform = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio))), antialias=True),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        # transforms.AugMix(),
        transforms.ToTensor(),
    ])
        
    print('strong augmentation', strong_transform)
        
    strong_transform2 = strong_transform

    val_transform = transforms.Compose([
        # transforms.Resize(math.floor(int(img_size / crop_ratio))),
        # transforms.CenterCrop(img_size),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
    ])
    
    thief_data = ImageNet_Base(dataset_root)
    indices = list(range(len(thief_data)))
    print(len(thief_data))
    random.shuffle(indices)

    # do not use the entire unlabeled set, use only SUBSET number of samples
    indices = indices[:args.subset]
    print(len(indices))
    unlabeled_set = indices

    if labeled_idxs is not None:
        labeled_set = labeled_idxs
        val_set = val_idxs
    else:
        num_train = args.num_labels*9//10
        labeled_set = indices[:num_train]
        val_set = indices[num_train:args.num_labels]

    thief_data_lb = ImageNet_Full(alg, dataset_root, labeled_set, target_model,  
                                  weak_transform=weak_transform)
    thief_data_val = ImageNet_Full(alg, dataset_root, val_set, target_model,  
                                  weak_transform=val_transform)
    thief_data_ulb = ImageNet_Full(alg, dataset_root, unlabeled_set, target_model, 
                                   is_ulb=True,
                                   weak_transform=weak_transform,
                                    # weak_transform=val_transform, 
                                   strong_transform=strong_transform, 
                                   strong_transform2=strong_transform2)
    
    if alg == 'selfkdcontrastive':
        weak_transform = transforms.Compose([
            transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            ])
        thief_data_ulb = ImageNet_Full(alg, dataset_root, unlabeled_set, target_model, 
                                   is_ulb=True,
                                   weak_transform=TwoCropTransform(weak_transform),
                                   strong_transform=strong_transform, 
                                   strong_transform2=strong_transform2)
        
    
    return thief_data_lb, thief_data_ulb, thief_data_val


class ImageNet_Full(ImageFolder, BasicDataset):
    test_frac = 0.0

    def __init__(self, alg, dataset_root, indexs, victim_model, 
                #  labeled=False, 
                 is_ulb=False,
                 weak_transform = None, strong_transform = None, 
                 strong_transform2 = None):
        
        self.alg = alg
        self.is_ulb = is_ulb
        self.transform = weak_transform
        self.strong_transform = strong_transform
        self.strong_transform2 = strong_transform2

        self.base_dataset = ImageNet_Base(dataset_root)
        subset_ds = Subset(self.base_dataset, indexs)
        subset_loader = DataLoader(subset_ds, batch_size=1000, num_workers=2, shuffle=False, drop_last=False, pin_memory=False)
        # self.labeled = labeled

        # query labels from victim model if labeled dataset
        # if labeled:
        if self.is_ulb is not True:
            victim_model.eval()
            self.samples = []
            with torch.no_grad():
                for d, l0, ind0 in tqdm(subset_loader):
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
        path, target = self.samples[index]
        img = self.base_dataset.loader(path)
        return img, target

    def __getitem__(self, index):
        return BasicDataset.__getitem__(self, index)
    
    def __len__(self):
        return len(self.samples)


class ImageNet_Base(ImageFolder):
    test_frac = 0.0

    def __init__(self, dataset_root, train=True, transform=None, target_transform=None):
        
        print('dataset root', dataset_root)
        if not osp.exists(dataset_root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                dataset_root, 'http://image-net.org/download-images'
            ))
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
        
        # Initialize ImageFolder
        super().__init__(dataset_root, transform=self.transform,
                         target_transform=target_transform)
        
        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))
        self.last_ok = -1
        self.num_corrupt = 0

        # # add an extra label placeholder in self.samples
        # for i in range(len(self.samples)):
        #     self.samples[i] = (self.samples[i][0], self.samples[i][1], self.samples[i][1])
            

    def __getitem__(self, index):
        path, target = self.samples[index]
        flag=True
        while(flag):
            try:
                sample = self.loader(path)
                flag=False
                self.last_ok = index
            except:
                import pdb;pdb.set_trace()
                self.num_corrupt = self.num_corrupt + 1
                path,target = self.samples[self.last_ok]
                
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


