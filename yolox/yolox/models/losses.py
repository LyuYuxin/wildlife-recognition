#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=15000.):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        # self.bcewithlog_loss = nn.BCEWithLogitsLoss(weight=self.weight, reduction="none")
        
    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - target),1) # N * 1
        feats = torch.sum(torch.abs(features), 1).reshape(-1)
        ib = grads * feats
        ib = self.alpha / (ib + self.epsilon)
        loss = F.cross_entropy(input, target, weight=self.weight, reduction='none') * ib
        loss = loss.mean()
        return loss


class EQloss(nn.Module):
    def __init__(self, freq_info, lambda_=0.02) -> None:
        super().__init__()
        self.freq_info = freq_info
        self.lambda_=lambda_
    # def exclude_func(self):
    #     # instance-level weight
    #     bg_ind = cls_nums
    #     weight = (self.gt_classes != bg_ind).float()
    #     weight = weight.view(roi_nums, 1).expand(roi_nums, cls_nums)
    #     return weight

    def threshold_func(self, input):
        # class-level weight
        roi_nums, cls_nums = input.size()
        weight = input.new_zeros(cls_nums)

        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, cls_nums).expand(roi_nums, cls_nums)
        return weight

    def forward(self, input, target):

        roi_nums, _ = input.size()

        eql_w = 1 - self.threshold_func(input) * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(input, target,
                                                        reduction='none')

        return torch.sum(cls_loss * eql_w) / roi_nums