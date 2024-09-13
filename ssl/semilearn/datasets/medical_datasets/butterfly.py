import os
import glob
import random
import math

from PIL import Image
import numpy as np
from skimage import color
import scipy.io as sio
from tqdm import tqdm
import pickle
from collections import OrderedDict, defaultdict

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler

from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def get_butterfly(args, alg, target_model, num_labels, num_classes, 
                       dataset_root, labeled_idxs=None, val_idxs=None):
    
    num_labels = num_labels // num_classes
    print('dataset root', dataset_root)

    img_size = args.img_size
    crop_ratio = args.crop_ratio

    if args.victim_arch == 'gbcnet':
        transforms1 = transforms.Compose([transforms.Resize((img_size, img_size)),\
                            transforms.ToTensor()])
    elif args.victim_arch == 'radformer':
        normalize = transforms.Normalize(  
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        transforms1 = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(), 
                                        normalize])
        # weak_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
        #                                 transforms.ToTensor(), 
        #                                 normalize])
        weak_transform = transforms.Compose([
                                transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
                                transforms.RandomCrop((img_size, img_size)),
                                transforms.ColorJitter(0.1, 0.1, 0.1, 0),                                
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize
                            ])
    elif args.victim_arch == 'resnet50':
        # normalize = transforms.Normalize(  
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        #     )
        transforms1 = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor()])
        # weak_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
        #                                 transforms.ToTensor(), 
        #                                 normalize])
        weak_transform = transforms.Compose([
                                transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
                                transforms.RandomCrop((img_size, img_size)),
                                transforms.ColorJitter(0.1, 0.1, 0.1, 0),                                
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ])
        
    elif args.victim_arch == 'resnet18':
        normalize = transforms.Normalize(  
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25]
            )
        transforms1 = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),normalize])
        
        weak_transform = transforms.Compose([
                                transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
                                transforms.RandomCrop((img_size, img_size)),
                                transforms.ColorJitter(0.1, 0.1, 0.1, 0),                                
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),normalize
                            ])
    
    val_transform = transforms1

    strong_transform = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio))), antialias=True),
        # RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0),
        transforms.RandomHorizontalFlip(),
        # RandAugment(3, 10),
        # transforms.AugMix(),
        transforms.ToTensor(),normalize
    ])

    thief_data = ButterflyDataset_Base(dataset_root)
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

    thief_data_lb = ButterflyDataset(alg, dataset_root, labeled_set, target_model,  
                                  weak_transform=weak_transform, 
                                  val_transform=val_transform)
    thief_data_val = ButterflyDataset(alg, dataset_root, val_set, target_model,  
                                  weak_transform=weak_transform, 
                                  val_transform=val_transform)
    thief_data_ulb = ButterflyDataset(alg, dataset_root, unlabeled_set, target_model, 
                                   is_ulb=True,
                                   weak_transform=weak_transform,
                                   strong_transform=strong_transform, 
                                   strong_transform2=strong_transform, 
                                   val_transform=val_transform)

    return thief_data_lb, thief_data_ulb, thief_data_val
    

class ButterflyDataset(BasicDataset):
    def __init__(self, alg, dataset_root, indexs, victim_model, 
                #  labeled=False, 
                 is_ulb=False, val_transform=None,
                 weak_transform = None, strong_transform = None, 
                 strong_transform2 = None):
        
        self.alg = alg
        self.is_ulb = is_ulb
        self.transform = weak_transform
        self.strong_transform = strong_transform
        self.strong_transform2 = strong_transform2
        # self.pickle_root=pickle_root

        self.base_dataset = ButterflyDataset_Base(dataset_root, transform=val_transform)
        subset_ds = Subset(self.base_dataset, indexs)
        subset_loader = DataLoader(subset_ds, batch_size=128, num_workers=4, shuffle=False, drop_last=False, pin_memory=False)

        # query labels from victim model if labeled dataset
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


class ButterflyDataset_Base(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, LabelList=None, DataList=None):
        """
        Ultrasound self-supervised training Dataset, only choose one image from a video
        :param data_dir: str
        :param transform: torch.transform
        """
        # self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.LabelList = LabelList
        self.DataList = DataList
        self.loader = pil_loader
        self.samples = self.get_img_info(data_dir)

    def __getitem__(self, index):
        sample, target = self.samples[index]  # list
        # print("index", self.samples[index])
        # path_img = random.sample(path_imgs, 1)  # random choose one image
        # img1 = Image.open(path_imgs).convert('RGB')  # 0~255
        # img2 = Image.open(path_imgs).convert('RGB')  # 0~255
        # label1 = 0 if path_img[0].lower()[64:].find("cov") > -1 else (1 if path_img[0].lower()[64:].find("pneu") > -1 else 2)

        # if self.transform is not None:
        #     img1, img2 = self.transform((img1, img2))  # transform

        if self.transform is not None:
            sample = self.loader(sample)
            sample = self.transform(sample)
        
        return sample, target, index

    def __len__(self):  # len
        return len(self.samples)
    
    @staticmethod
    def get_img_info(data_dir):
        # data_info = list()
        samples = []
        video_count = 0
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:  # one video as one class
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), img_names))
                path_imgs = []
                # print("sub", sub_dir)
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    path_imgs.append(path_img)
                    samples.append((path_img, video_count))
                # data_info.append(path_imgs)
                video_count += 1

        # print(samples)
        return samples