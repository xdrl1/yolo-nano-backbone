#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "/home/vitis-ai-user/yolonano/bbd_data"
        self.train_ann = "/home/vitis-ai-user/yolonano/code/datasets/COCO/annotations/instances_train2017.json"
        self.val_ann = "/home/vitis-ai-user/yolonano/code/datasets/COCO/annotations/instances_val2017.json"
        self.train_img_dir = "/home/vitis-ai-user/yolonano/bbd_data/bbd2k5-images-image"
        self.val_img_dir = "/home/vitis-ai-user/yolonano/bbd_data/bbd2k5-images-image"
        self.name = "images"

        self.input_size = (2500, 2500)
        self.test_size = (2500, 2500)
        self.num_classes = 1
        self.data_num_workers = 1
        self.multiscale_range = 0
        self.max_epoch = 160
        self.batch_size = 4

        self.warmup_epochs = 10
        self.warmup_lr = 0.0001
        self.min_lr_ratio = 0.03
        self.basic_lr_per_img = 0.01 / 16.0
        self.test_conf = 0.04
        self.nmsthre = 0.25
        self.act = "relu"

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            # from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            from yolox.models import YOLOX, YOLOXHead
            from yolox.models.yolo_pafpn_deploy import YOLOPAFPN
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
