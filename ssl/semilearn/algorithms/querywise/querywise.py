# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F


@ALGORITHMS.register('querywise')
class QueryWise(AlgorithmBase):

    """
        QueryWise algorithm

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, 
                   victim_model=None, labeled_set=None, val_set=None, test_loader=None):
        super().__init__(args, net_builder, tb_log, logger,
                         victim_model, labeled_set, val_set, test_loader) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

        # logit adjustment
        self.la = self.args.la
        if self.la is True:
            self.compute_adjustment(self.args.tro)
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()


    def compute_adjustment(self, tro):
        """compute the base probabilities for logit adjustment"""
        lb_class_dist = [0 for _ in range(self.num_classes)]
        loader = DataLoader(self.dataset_dict['train_lb'], 
                            sampler=SequentialSampler(self.dataset_dict['train_lb']),
                            batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                            drop_last=False)
        for _, data in enumerate(loader):
            target = data['y_lb']
            for j in target:
                key = int(j.item())
                lb_class_dist[key] += 1

        lb_class_dist = np.array(lb_class_dist)
        label_freq_array = lb_class_dist / lb_class_dist.sum()

        adjustments = np.log(label_freq_array ** tro + 1e-12)
        adjustments = torch.from_numpy(adjustments).float()
        self.adjustments = adjustments.cuda()
        print('logit adjustments: ', self.adjustments)


    def anchor_init(self, pretrained_state_dict=None):
        # initialize anchor model
        self.anchor_model = self.net_builder(num_classes=self.num_classes)
        self.anchor_model.load_state_dict(pretrained_state_dict)
        self.anchor_model = self.anchor_model.cuda()

        self.anchor_params = {}
        # self.model.eval()
        for name, param in self.model.named_parameters():
            # if param.requires_grad:
            self.anchor_params[name] = param.data.clone()


    def stochastic_restore(self):
        for nm, m  in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.args.rst).float().cuda() 
                    with torch.no_grad():
                        name = f"{nm}.{npp}"
                        assert name in self.anchor_params

                        p.data = self.anchor_params[name].cuda() * mask + p * (1-mask)


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
            
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs[num_lb:].chunk(2)
            else:
                logits_x_lb = outs_x_ulb_w
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w

            if self.la is True:
                logits_x_lb = logits_x_lb + self.adjustments

            # SUPERVISED LOSS (on labeled data)

            # obtain soft labels from anchor model
            self.anchor_model.eval()
            with torch.no_grad():
                y_lb_soft = self.anchor_model(x_lb)
            
            # knowledge distillation loss
            loss_ce =  self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            loss_kd = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits_x_lb / self.args.kd_temp, dim=1),  # input
                                                                F.softmax(y_lb_soft / self.args.kd_temp, dim=1))        # target
            sup_loss = (1. - self.args.kd_alpha) * loss_ce \
                        + self.args.kd_alpha * self.args.kd_temp * self.args.kd_temp * loss_kd

            # UNSUPERVISED LOSS (on unlabeled data)

            # compute pseudolabels using teacher model
            self.ema.apply_shadow()
            with torch.no_grad():
                logits_x_ulb_w_teacher = self.model(x_ulb_w)
            self.ema.restore()

            # compute pseudolabels using anchor model
            self.anchor_model.eval()
            with torch.no_grad():
                logits_x_ulb_w_anchor = self.anchor_model(x_ulb_w)

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask using anchor MSPs
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w_anchor, softmax_x_ulb=True)

            # knowledge distillation loss from teacher
            unsup_loss_kd_teacher = self.args.kd_temp_ulb * self.args.kd_temp_ulb * \
                                    torch.nn.KLDivLoss(reduction='none')(F.log_softmax(logits_x_ulb_w  / self.args.kd_temp_ulb, dim=1),  # input
                                                                F.softmax(logits_x_ulb_w_teacher / self.args.kd_temp_ulb, dim=1))       # target
            
            # knowledge distillation loss from anchor
            unsup_loss_kd_anchor = self.args.kd_temp_ulb * self.args.kd_temp_ulb * \
                                    torch.nn.KLDivLoss(reduction='none')(F.log_softmax(logits_x_ulb_w / self.args.kd_temp_ulb, dim=1),  # input
                                                                F.softmax(logits_x_ulb_w_anchor / self.args.kd_temp_ulb, dim=1))       # target
                        
            # weighted unlabeled loss
            unsup_loss = (1. - self.args.kd_alpha_ulb) * unsup_loss_kd_teacher.sum(dim=-1) \
                        + self.args.kd_alpha_ulb * unsup_loss_kd_anchor.sum(dim=-1)
            unsup_loss = (unsup_loss * mask).mean()

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
