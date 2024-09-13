from __future__ import print_function, division
import torch
import numpy as np

import torch.utils
from conf import cfg, load_cfg_fom_args
from loader_utils import *

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pred_label(pred_tensor):
    _, pred = torch.max(pred_tensor, dim=1)
    return pred.item()


def validate(model, val_loader, victim_arch):
    model.eval()
    y_true, y_pred = [], []
    for i, (inp, target, fname) in enumerate(val_loader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target)

            if len(input_var.shape) == 5:
                images = input_var.squeeze(0)
                outputs =  model(images)
                _, pred = torch.max(outputs, dim=1)
                pred_label = torch.max(pred)
                pred_label = pred_label.unsqueeze(0)
                
                y_true.append(target_var.tolist()[0][0])
                y_pred.append(pred_label.item())

            else:
                outputs = model(input_var)
                if victim_arch == 'radformer':
                    g_out, l_out, f_out, _ = outputs

                y_true.append(target_var.tolist()[0])
                _, pred_label = torch.max(f_out, dim=1)
                y_pred.append(pred_label.item()) 
            
    return y_true, y_pred

    
if __name__ == "__main__":

    load_cfg_fom_args(description='Model Stealing')
        
    # Load victim dataset (test split only)
    testset, test_loader, n_classes = load_victim_dataset(cfg, cfg.VICTIM.DATASET)
    print(f"Loaded target dataset of size {len(testset)} with {n_classes} classes")
    
    # Load victim model    
    target_model = load_victim_model(cfg.VICTIM.ARCH, cfg.VICTIM.PATH)

    # Evaluate target model on target dataset
    y_true, y_pred = validate(target_model, test_loader, cfg.VICTIM.ARCH)
    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    sens = cfm[2][2]/np.sum(cfm[2])
    spec = ((cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1])))
    acc_binary = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1] + cfm[2][2])/(np.sum(cfm))
    print("Acc-3class: %.4f Acc-Binary: %.4f Specificity: %.4f Sensitivity: %.4f"%(acc, acc_binary, spec, sens))

    print('specificity = {}/{}'.format(cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1], np.sum(cfm[0]) + np.sum(cfm[1])))
    print('sensitivity = {}/{}'.format(cfm[2][2], np.sum(cfm[2])))
    