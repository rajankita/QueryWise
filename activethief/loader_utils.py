from tqdm import tqdm
import os, sys

import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, resnet18, vit_b_16
from inception import inception_v3

sys.path.append('GBCNet')
from GBCNet.models import GbcNet

sys.path.append('Radformer')
from RadFormer.models import RadFormer
from RadFormer.dataloader import GbUsgDataSet

from covid_dataset import COVIDDataset
from covidx_dataset import COVIDxDataset


class Victim(nn.Module):
    """class for victim model

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model, arch):
        super(Victim, self).__init__()
        self.model = model
        self.arch = arch
        
    def forward(self, x):
        # change forward here
        x = torch.nn.functional.interpolate(x, size=224)
        out = self.model(x)
        return out


def load_victim_model(arch, model_path):
    # Define architecture
    if arch == 'gbcnet':
        target_model = GbcNet(num_cls=3, pretrain=False)
    elif arch == 'radformer':
        target_model = RadFormer(local_net='bagnet33', \
                        num_cls=3, \
                        global_weight=0.55, \
                        local_weight=0.1, \
                        fusion_weight=0.35, \
                        use_rgb=True, num_layers=4, pretrain=False)
    elif arch == 'resnet18':
        target_model = resnet18(num_classes=3) 
    
    # Load weights
    print('target model keys: ', len(target_model.state_dict().keys()))
    checkpoint_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint_dict:
        checkpoint_dict = checkpoint_dict['state_dict']
    print('checkpoint keys: ', len(checkpoint_dict.keys()))
    target_model.load_state_dict(checkpoint_dict, strict=True)
    # target_model.net = target_model.net.float().cuda()
    target_model = Victim(target_model.float().cuda(), arch)
    
    
    return target_model


def load_thief_model(cfg, arch, n_classes, pretrained_path, load_pretrained=True):

    pretrained_state = torch.load(pretrained_path) 
    
    if arch == 'resnet18':
        thief_model = resnet18(num_classes=n_classes)
    elif arch == 'resnet34':
        thief_model = resnet34(num_classes=n_classes)
    elif arch == 'resnet50':
        thief_model = resnet50(num_classes=n_classes)
    elif arch == 'deit':
        thief_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', 
                                     pretrained=False, num_classes=n_classes)
        pretrained_state = pretrained_state['model']

    elif arch == 'vit':
        thief_model = vit_b_16(num_classes=n_classes)
    
    elif arch == 'inception_v3':
        thief_model = inception_v3(num_classes=n_classes)

    if load_pretrained == True:
        thief_state = thief_model.state_dict()
        print('thief state: ', print(thief_state.keys()))
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in thief_state and v.size() == thief_state[k].size() }
        print('pretrained state: ', pretrained_state.keys())
        thief_state.update(pretrained_state)
        thief_model.load_state_dict(thief_state, strict=True)
    thief_model = thief_model.cuda()
    
    return thief_model


def load_victim_dataset(cfg, dataset_name):
    
    if dataset_name == 'gbusg':
        n_classes = 3
        set_dir=cfg.VICTIM.DATA_ROOT
        img_dir=os.path.join(set_dir, 'imgs') 
        list_file = os.path.join(set_dir, 'test.txt')
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        testset = GbUsgDataSet(data_dir=img_dir, 
                                image_list_file=list_file,
                                transform=transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        test_loader = DataLoader(dataset=testset, batch_size=32, 
                                shuffle=False, num_workers=4)

    elif dataset_name == 'pocus':
        n_classes = 3
        valid_transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
                            ])

        img_dir = os.path.join(cfg.VICTIM.DATA_ROOT, 'covid_data1.pkl')
        testset = COVIDDataset(data_dir=img_dir, 
                               train=False, 
                               transform=valid_transform)

        test_loader = DataLoader(dataset=testset, batch_size=32, 
                                    shuffle=False, num_workers=4)

    return testset, test_loader, n_classes

   
def load_thief_dataset(cfg, dataset_name, data_root, target_model):
   
    if dataset_name == 'GBUSV':
        from gbusv_dataset import GbVideoDataset

        # Create an instance of the custom dataset
        if cfg.VICTIM.ARCH == 'gbcnet':
            transforms1 = transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.HEIGHT)),\
                                transforms.ToTensor()])
        elif cfg.VICTIM.ARCH == 'radformer':
            normalize = transforms.Normalize(  
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
            transforms1 = transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.WIDTH)),
                                            transforms.ToTensor(), 
                                            normalize])
            transforms2= transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.WIDTH)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandAugment(),
                                            transforms.ToTensor(),
                                            normalize
                                            ])
        
        thief_data = GbVideoDataset(data_root, transforms1)
                                    # pickle_root='/home/deepankar/scratch/MSA_Medical/')
        thief_data_aug = GbVideoDataset(data_root, transforms2)
                                        # pickle_root='/home/deepankar/scratch/MSA_Medical/')
        
    elif dataset_name == 'covidx':

        normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
        
        transforms1 = transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.WIDTH)),
                                        transforms.ToTensor(), 
                                        normalize
                                        ])
        transforms2= transforms.Compose([transforms.Resize((cfg.VICTIM.WIDTH, cfg.VICTIM.WIDTH)),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.RandAugment(),
                                        transforms.ToTensor(), 
                                        normalize
                                        ])
        
        thief_data = COVIDxDataset(data_root, transforms1)
        thief_data_aug = COVIDxDataset(data_root, transforms2)
        
    else:
        raise AssertionError('invalid thief dataset')
    
    return thief_data, thief_data_aug
        
    
def create_thief_loaders(thief_data, thief_data_aug, labeled_set, val_set, unlabeled_set, batch_size, target_model):
    
    print("replacing labeled set labels with victim labels")
    # print(labeled_set)
    thiefdataset = Subset(thief_data, labeled_set)
    train_loader = DataLoader(thiefdataset, batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=False)
    target_model.eval()
    list1=[]
    with torch.no_grad():
        for d, l0, ind0 in tqdm(train_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            for ii, jj in enumerate(ind0):
                thief_data_aug.samples[jj] = (thief_data_aug.samples[jj][0], l[ii])
                list1.append((jj.cpu().tolist(), l[ii]))
        
    train_loader = DataLoader(Subset(thief_data_aug, labeled_set), batch_size=batch_size,
                            pin_memory=False, num_workers=4, shuffle=True)
    # unlabeled_loader = DataLoader(Subset(thief_data_aug, unlabeled_set), batch_size=batch_size, 
    #                                     pin_memory=False, num_workers=4, shuffle=True)
    unlabeled_loader = None
    
    print("replacing val labels with victim labels")
    val_loader = DataLoader(Subset(thief_data, val_set), batch_size=batch_size, 
                            pin_memory=False, num_workers=4, shuffle=True)
    target_model.eval()
    with torch.no_grad():
        for d,l,ind0 in tqdm(val_loader):
            d = d.cuda()
            l = target_model(d).argmax(axis=1, keepdim=False)
            l = l.detach().cpu().tolist()
            # print(l)
            for ii, jj in enumerate(ind0):
                thief_data.samples[jj] = (thief_data.samples[jj][0], l[ii])
            
                
    return train_loader, val_loader, unlabeled_loader, list1