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


def get_gbvideodataset(args, alg, target_model, num_labels, num_classes, 
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
<<<<<<< HEAD
        # weak_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
        #                                     transforms.RandomHorizontalFlip(),
        #                                     transforms.RandAugment(),
        #                                     transforms.ToTensor(),
        #                                     normalize
        #                                     ])
        print('weak transform', weak_transform)
=======
    
    # weak_transform = transforms1
    val_transform = transforms1
>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d

    val_transform = transforms1
    strong_transform = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio))), antialias=True),
        # RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0),
        transforms.RandomHorizontalFlip(),
        # RandAugment(3, 10),
        # transforms.AugMix(),
        transforms.ToTensor(),
        normalize
    ])

    thief_data = GbVideoDataset_Base(dataset_root)
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

    thief_data_lb = GBVideoDataset(alg, dataset_root, labeled_set, target_model,  
                                  weak_transform=weak_transform, 
<<<<<<< HEAD
                                  val_transform=val_transform,
                                #   pickle_root='/home/deepankar/scratch/MSA_Medical/'
                                  )
    thief_data_val = GBVideoDataset(alg, dataset_root, val_set, target_model,  
                                  weak_transform=weak_transform, 
                                  val_transform=val_transform,
                                #   pickle_root='/home/deepankar/scratch/MSA_Medical/'
                                  )
=======
                                  val_transform=val_transform,pickle_root='/home/deepankar/scratch/MSA_Medical/')
    thief_data_val = GBVideoDataset(alg, dataset_root, val_set, target_model,  
                                  weak_transform=weak_transform, 
                                  val_transform=val_transform,pickle_root='/home/deepankar/scratch/MSA_Medical/')
>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d
    thief_data_ulb = GBVideoDataset(alg, dataset_root, unlabeled_set, target_model, 
                                   is_ulb=True,
                                   weak_transform=weak_transform,
                                   strong_transform=strong_transform, 
                                   strong_transform2=strong_transform, 
<<<<<<< HEAD
                                   val_transform=val_transform,
                                #    pickle_root='/home/deepankar/scratch/MSA_Medical/'
                                   )
=======
                                   val_transform=val_transform,pickle_root='/home/deepankar/scratch/MSA_Medical/')
>>>>>>> ba1060430d61521041e28b47a409dc50ce80da3d

    return thief_data_lb, thief_data_ulb, thief_data_val



class GBVideoDataset(BasicDataset):
    def __init__(self, alg, dataset_root, indexs, victim_model, 
                #  labeled=False, 
                 is_ulb=False, val_transform=None,
                 weak_transform = None, strong_transform = None, 
                 strong_transform2 = None,pickle_root=None):
        
        self.alg = alg
        self.is_ulb = is_ulb
        self.transform = weak_transform
        self.strong_transform = strong_transform
        self.strong_transform2 = strong_transform2
        self.pickle_root=pickle_root

        self.base_dataset = GbVideoDataset_Base(dataset_root,pickle_root, transform=val_transform)
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


class GbVideoDataset_Base(torch.utils.data.Dataset):
    def __init__(self, root, pickle_root=None, transform=None, return_all_video_frames=True, data_split='all'):
        self.transform = transform
        self.root = root
        self.transform = transform
        self.return_all_video_frames = return_all_video_frames ## True
        self.data_split = data_split
        self.pickle_root = pickle_root
        self._get_annotations()

        self.loader = pil_loader

    @staticmethod
    def get_video_name(name):
        return name.split("/")[-2]

    @staticmethod
    def get_frame_id(name):
        return int(name.split("/")[-1][:-4])

    def get_image_paths(self):
        print('path ############', self.data_basepath)
        return sorted(list(tqdm(glob.iglob(os.path.join(self.data_basepath, "*/*.jpg")))))

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.data_split_path, key,  "%05d.jpg" % ind)

    def video_id_frame_id_split(self, name):
        return self.get_video_name(name), self.get_frame_id(name)

    def _get_single_frame(self, path_key, ind):
        return self.transform(self.loader(self.get_image_name(path_key, ind)))

    def _get_annotations(self):
        self.data_basepath = self.root
        self.data_split_path = os.path.join(self.data_basepath)

        # create a flattened list of all image paths
        if self.pickle_root is not None:
            pickle_path = os.path.join(self.pickle_root, self.data_split+ "_names.pkl")
        else:
            pickle_path = os.path.join(self.data_basepath, self.data_split+ "_names.pkl")
        if not os.path.exists(pickle_path):
            print('creat new cache')
            images = self.get_image_paths()
            samples = []
            video_names = []
            video_count = 0
            video_frames = sorted([self.video_id_frame_id_split(name) for name in images])
            for vid_id, ind in video_frames:
                if vid_id not in video_names:
                    video_names.append(vid_id)
                    video_count += 1
                path = self.get_image_name(vid_id, ind)
                label = video_count - 1
                samples.append((path, label))
            pickle.dump(samples, open(pickle_path, "wb"))
        self.samples = pickle.load(open(pickle_path, "rb"))
        print("Num of videos %d frames %d" % (len(set([e[1] for e in self.samples])), len(self.samples)))


    def __getitem__(self, index):
        path, target = self.samples[index]

        ## Loading the image at the chosen index 
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target, index
        

    def __len__(self):
        return len(self.samples)
