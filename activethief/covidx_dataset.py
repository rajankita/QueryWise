import os
from PIL import Image
import torch

"""
source: https://github.com/983632847/USCL/blob/main/train_USCL/data_aug/dataset_wrapper_Ultrasound_Video_Mixup.py
"""
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
    
class COVIDxDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, LabelList=None, DataList=None):
        """
        Ultrasound self-supervised training Dataset, only choose one image from a video
        :param data_dir: str
        :param transform: torch.transform
        """
        self.transform = transform
        self.LabelList = LabelList
        self.DataList = DataList
        self.loader = pil_loader
        self.samples = self.get_img_info(data_dir)
        self.loader = pil_loader


    def __getitem__(self, index):
        sample, target = self.samples[index]  # list
        if self.transform is not None:
            sample = self.loader(sample)
            sample = self.transform(sample)
        
        return sample, target, index

    def __len__(self):  # len
        return len(self.samples)
    
    @staticmethod
    def get_img_info(data_dir):
        samples = []
        img_names = os.listdir(data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), img_names))
        path_imgs = []
        for i in range(len(img_names)):
            img_name = img_names[i]
            path_img = os.path.join(data_dir, img_name)
            path_imgs.append(path_img)
            samples.append((path_img, 0))

        return samples