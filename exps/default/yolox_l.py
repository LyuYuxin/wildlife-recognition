#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # Define yourself dataset path
        self.data_dir = "/home/lyx/Codes/New Data/"
        self.train_ann = "train_annotations.json"
        self.val_ann = "val_annotations.json"

        self.num_classes = 9

        self.max_epoch = 100
        self.data_num_workers = 8
        self.eval_interval = 1
        self.input_size = (640, 640)