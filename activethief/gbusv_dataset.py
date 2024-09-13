import os
import glob
from PIL import Image
import tqdm
import pickle
import torch


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
    

class GbVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_all_video_frames=True, data_split='all', pickle_root=None):
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
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.data_basepath, "*/*.jpg")))))
    
    def get_image_paths_benign(self):
        print('path ############', self.data_basepath)
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.data_basepath, "benign*/*.jpg")))))

    def get_image_paths_malignant(self):
        print('path ############', self.data_basepath)
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.data_basepath, "malignant*/*.jpg")))))

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
        # pickle_path = os.path.join(self.data_basepath, "all_paths.pkl")
        if self.pickle_root is not None:
            pickle_path = os.path.join(self.pickle_root, self.data_split+ "_names.pkl")
        else:
            pickle_path = os.path.join(self.data_basepath, self.data_split+ "_names.pkl")
        print(pickle_path)
        if not os.path.exists(pickle_path):
            print('create new cache')
            if self.data_split == 'all':
                images = self.get_image_paths()
            elif self.data_split == 'benign':
                images = self.get_image_paths_benign()
            elif self.data_split == 'malignant':
                images = self.get_image_paths_malignant()
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
        if self.transform is not None:
            sample = self.loader(path)
            sample = self.transform(sample)
        
        return sample, target, index
        

    def __len__(self):
        return len(self.samples)