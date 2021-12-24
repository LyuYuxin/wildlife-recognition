#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from cv2 import RETR_CCOMP
import torch.nn as nn
from torch.nn.modules.module import T
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # Define yourself dataset path
        self.data_dir = "/home/lyx/Codes/New Data/"
        
        # self.train_ann = "full_train_annotations.json" # 1.0 portion
        # self.val_ann = "full_val_annotations.json"
        self.train_ann = "0.1_train_annotations.json"# 0.1 portion
        self.val_ann = "0.1_val_annotations.json"
        # self.train_ann = "luxin_train.json"
        # self.val_ann = "luxin_val.json"
        self.num_classes = 2
        self.no_aug_epochs = 20


        self.ema = False
        self.max_epoch = 300
        self.data_num_workers = 8
        self.eval_interval = 1
        self.input_size = (768, 768)
        # self.test_size = (768, 768)

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            #tag lyx 
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=False, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, exp=self)

            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_cls_num_list(self):
        return self.dataset._dataset.cls_nums_list