import logging
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn.parallel
from torchvision import transforms as T
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import resnet18, resnet50

sys.path.append('/home/ankita/scratch/MSA_Medical/')
sys.path.append('/home/ankita/scratch/MSA_Medical/GBCNet')
from GBCNet.models import GbcNet

sys.path.append('RadFormer')
from RadFormer.models import RadFormer
from RadFormer.dataloader import GbUsgDataSet

logger = logging.getLogger(__name__)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    

def create_model(arch, num_classes):
    
    if arch == 'resnet18':
        model = resnet18(num_classes=num_classes) 
    elif arch == 'resnet50':
        model = resnet50(num_classes=num_classes)  
    elif arch == 'gbcnet':
        model = GbcNet(num_cls=3, pretrain=False)
    elif arch == 'radformer':
        model = RadFormer(local_net='bagnet33', \
                        num_cls=3, \
                        global_weight=0.55, \
                        local_weight=0.1, \
                        fusion_weight=0.35, \
                        use_rgb=True, num_layers=4, pretrain=False)
    
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    
    return model


def test(args, test_loader, model, verbose=True):

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            input_var = torch.autograd.Variable(data[0].cuda())
            target_var = torch.autograd.Variable(data[1])

            if len(input_var.shape) == 5:
                images = input_var.squeeze(0)
                outputs =  model(images)
                _, pred = torch.max(outputs, dim=1)
                pred_label = torch.max(pred)
                pred_label = pred_label.unsqueeze(0)
                
                y_true.append([target_var.tolist()[0][0]])
                y_pred.append([pred_label.tolist()])

            else:
                outputs = model(input_var)
                _, pred_label = torch.max(outputs, dim=1)
                y_pred.append(pred_label.tolist()) 
                y_true.append(target_var.tolist())

        if not args.no_progress:
            test_loader.close()
    
    y_pred = np.concatenate(y_pred, 0)
    y_true = np.concatenate(y_true, 0)

    accuracy_per_class = classwise_accuracy(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2])
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    if verbose == True:
        print('specificity = {}/{}'.format(cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1], np.sum(cfm[0]) + np.sum(cfm[1])))
        print('sensitivity = {}/{}'.format(cfm[2][2], np.sum(cfm[2])))

    logger.info("acc: {:.2f}".format(acc))

    
    return acc, spec, sens, accuracy_per_class


def classwise_accuracy(true_labels, predicted_labels):
    # Check if the input arrays have the same length
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Input arrays must have the same length.")

    # Create a confusion matrix
    confusion_matrix = np.zeros((3, 3), dtype=int)

    # Populate the confusion matrix
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Compute classwise accuracy
    classwise_accuracy = np.zeros(3)
    for i in range(3):
        if np.sum(confusion_matrix[i, :]) != 0:
            classwise_accuracy[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

    return classwise_accuracy


def test_thief(model, net, victim_model, test_loader, ema=False,  out_key='logits'):
    net.eval()
    if ema is True:
        model.ema.apply_shadow()
    victim_model.eval()

    total_num = 0.0
    total_matches = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            input_var = torch.autograd.Variable(data[0].cuda())
            target_var = torch.autograd.Variable(data[1])

            num_batch = target_var.shape[0]
            total_num += num_batch
            
            if len(input_var.shape) == 5:
                images = input_var.squeeze(0)
                # logits =  net(images)['logits']
                logits =  net(images)
                _, pred = torch.max(logits, dim=1)
                pred_label = torch.max(pred)
                pred_label = pred_label.unsqueeze(0)
                logits_victim = victim_model(images)
                y_pred.append([pred_label.tolist()])
                y_true.append([target_var.tolist()[0][0]])

            else:
                # logits = net(input_var)['logits']
                logits = net(input_var)

                _, pred_label = torch.max(logits, dim=1)
                logits_victim = victim_model(input_var)
                y_pred.extend(pred_label.tolist()) 
                y_true.extend(target_var.tolist())
            
            # y_true.extend(target_var.cpu().tolist())
            x1 = torch.max(logits, dim=-1)[1]
            x2 = torch.max(logits_victim, dim=-1)[1]
            # y_pred.extend(x1.cpu().tolist())
            total_matches += num_batch - int((torch.count_nonzero(x1-x2)).detach().cpu())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    agreement = total_matches / total_num
    cfm = confusion_matrix(y_true, y_pred)
    spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2])
    accuracy_per_class = classwise_accuracy(y_true, y_pred)

    if ema is True:
        model.ema.restore()
    net.train()
    
    return acc, agreement, spec, sens, accuracy_per_class


def agree(model1, model2, test_loader):
    c=0
    l=0
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            n=inputs.shape[0]
            x1=model1(inputs).argmax(axis=-1,keepdims=False)

            try:
                x2=model2(inputs).argmax(axis=-1,keepdims=False)
            except:
                x1=model1(inputs).argmax(axis=-1,keepdims=False)
                # x2=model2(inputs)['logits'].argmax(axis=-1,keepdims=False)
            c+=n-int((torch.count_nonzero(x1-x2)).detach().cpu())
            l+=n
            # print(c, l)
    print('Agreement between Copy and source model is ', c/l)
    return c / l


def dist(indices, dataloader):
    "Return label distribution of selected samples" 
    dl = dataloader
    d = {}
    print('Number of samples ', len(indices))
    labels = []
    with torch.no_grad():
        for data in (dl):
            label = data['y_lb']
            labels.extend(label.cpu().detach().numpy())
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        d[int(lbl)] = 0
    for label in labels:
        d[int(label)]+=1
    return d


def get_cosine_scheduler_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    # num_cycles=12./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def load_victim_dataset(args):

    if args.victim_dataset == 'gbusg':
        args.num_classes = 3
        set_dir=args.victim_data_root
        img_dir=os.path.join(set_dir, 'imgs') 
        list_file = os.path.join(set_dir, 'test.txt')
        normalize = T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        testset = GbUsgDataSet(data_dir=img_dir, 
                                image_list_file=list_file,
                                transform=T.Compose([
                                    T.Resize(224),
                                    T.CenterCrop(224),
                                    T.ToTensor(),
                                    normalize,
                                ]))
        
    elif args.victim_dataset == 'pocus':
        from covid_dataset import COVIDDataset
        valid_transform = T.Compose([T.Resize((224, 224)),
                    T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
                    ])

        args.num_classes = 3
        img_dir = os.path.join(args.victim_data_root, 'covid_data1.pkl')
        testset = COVIDDataset(data_dir=img_dir, 
                               train=False, 
                               transform=valid_transform)

    else:
        raise NotImplementedError
    
    return testset