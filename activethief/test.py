import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.utils
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from conf import cfg, load_cfg_fom_args

sys.path.append('/home/ankita/scratch/MSA_Medical')
from activethief.train_utils import agree
from conf import cfg, load_cfg_fom_args
from loader_utils import *


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def testz(model, dataloader, no_roi=True, verbose=True, logits=False, criterion=torch.nn.CrossEntropyLoss()):
    
    model.eval()
    y_true, y_pred = [], []
    softmaxes = []
    losses = AverageMeter()
    for i, (inp, target, fname) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target)
            outputs = model(input_var)

            # if soft labels, extract hard label from it
            if len(target_var.shape) > 1:
                target_var = target_var.argmax(axis=1)
            loss = criterion(outputs, target_var.cuda())
            losses.update(loss.item(), input_var.size(0))

            _, pred_label = torch.max(outputs, dim=1)
            y_pred.append(pred_label.tolist()) 
            softmaxes.extend(np.asarray(outputs.cpu()))
            
            y_true.append(target_var.tolist())

    y_pred = np.concatenate(y_pred, 0)
    y_true = np.concatenate(y_true, 0)
    softmaxes = np.asarray(softmaxes)
    print('y_pred ', y_pred.shape)
    print('y_true ', y_true.shape)
    print('softmaxes ', softmaxes.shape)

    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2])
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    eces = ECELoss().loss(softmaxes, y_true, n_bins=15, logits=logits)
    cces = SCELoss().loss(softmaxes, y_true, n_bins=15, logits=logits)

    if verbose == True:
        print('specificity = {}/{}'.format(cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1], np.sum(cfm[0]) + np.sum(cfm[1])))
        print('sensitivity = {}/{}'.format(cfm[2][2], np.sum(cfm[2])))
    
    return acc, f1, spec, sens, eces, cces, losses.avg


def compute_pseudolabel_acc(target_model, thief_model, thief_dataset, p_cutoff):
    
    dataloader = DataLoader(thief_dataset, batch_size=256,
                        pin_memory=False, num_workers=4, shuffle=False)
    target_model.eval()
    thief_model.eval()
    y_true, y_pred = [], []
    for (inp, target, fname) in tqdm(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target)

            outputs_thief = thief_model(input_var)
            _, pred_thief = torch.max(outputs_thief, dim=1)
            probs_thief = torch.softmax(outputs_thief, dim=-1)
            conf_thief, _ = torch.max(probs_thief.detach(), dim=-1)
            # y_pred.append(pred_thief.tolist()) 

            outputs_target = target_model(input_var)
            _, pred_target = torch.max(outputs_target, dim=1)
            # y_true.append(pred_target.tolist()) 

            for a, b, c in zip(conf_thief, pred_thief.tolist(), pred_target.tolist()):
                if a > p_cutoff:
                    y_pred.append(b)
                    y_true.append(c)

    # y_pred = np.concatenate(y_pred, 0)
    # y_true = np.concatenate(y_true, 0)

    acc = accuracy_score(y_true, y_pred)
    
    return acc


if __name__ == "__main__":

    load_cfg_fom_args(description='Model Stealing')
    thief_model_dir = '/home/ankita/mnt/data_msa_medical/results_ankita/gbusg_radformer/GBUSV_resnet50/SGD/5000_val500/random_v8/'

    trial = 1
    cycle = 1
    
    # Load victim dataset (test split only)
    testset, test_loader, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET)
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")

    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH)

    # Evaluate target model on target dataset: sanity check
    target_model.eval()
    acc, f1, spec, sens, ece, cce,_ = testz(target_model, test_loader, no_roi=False, logits=False)
    print('Target model Acc: {:.4f} Spec: {:.4f} Sens: {:.4f} ECE {:.4f} SCE {:.4f}'\
            .format(acc, spec, sens, ece, cce))

    # Load trained thief model
    thief_model_path = os.path.join(thief_model_dir, f'trial_{trial}_cycle_{cycle}_best.pth')

    thief_model = load_thief_model(cfg, cfg.THIEF.ARCH, n_classes, cfg.ACTIVE.PRETRAINED_PATH, load_pretrained=False)
    thief_state = thief_model.state_dict()
    print("Load thief model weights")
    pretrained_state = torch.load(thief_model_path) 
    if 'state_dict' in pretrained_state:
        pretrained_state = pretrained_state['state_dict']
    pretrained_state_common = {}
    for k, v in pretrained_state.items():
        if k in thief_state and v.size() == thief_state[k].size():
            pretrained_state_common[k] = v
        elif 'backbone.'+k in thief_state and v.size() == thief_state['backbone.'+k].size():
            pretrained_state_common['backbone.'+k] = v
        # remove 'module.' from pretrained state dict
        elif k[7:] in thief_state and v.size() == thief_state[k[7:]].size():
            pretrained_state_common[k[7:]] = v
        # remove 'base_model.' from pretrained state dict
        elif k[11:] in thief_state and v.size() == thief_state[k[11:]].size():
            pretrained_state_common[k[11:]] = v
        else:
            print('key not found', k)

    assert(len(thief_state.keys()) == len(pretrained_state_common.keys()))
    thief_state.update(pretrained_state_common)
    thief_model.load_state_dict(thief_state, strict=True)
    thief_model = thief_model.cuda()

    # Compute accuracy and agreement on test dataset
    print('Thief model')
    thief_model.eval()
    acc, f1, spec, sens, ece, cce, _ = testz(thief_model, test_loader, logits=True)
    agr = agree(target_model, thief_model, test_loader)
    print(f'Thief model on target dataset: acc = {acc:.4f}, agreement = {agr:.4f}, \
          f1 = {f1:.4f}, spec = {spec:.4f}, sens = {sens:.4f}, ECE {ece:.4f}, SCE {cce:.4f}')
            
    