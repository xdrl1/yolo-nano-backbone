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

        # self.random_size = (10, 20)
        self.data_dir = './drone2021_copy/'
        self.train_ann = "split_train_coco.json"
        self.val_ann = "split_val_coco.json"
        self.test_ann = "split_val_coco.json"
        self.coco_folder_name = "images"
        self.name="images"
        self.input_size = (2176, 3840)
        self.test_size = (2176, 3840)

        self.num_classes = 1
        self.data_num_workers = 1
        self.multiscale_range = 0
        self.max_epoch = 160
        
        
        # modified for 2 training
        self.warmup_epochs = 10 #increase warmup to 10
        # minimum learning rate during warmup
        self.warmup_lr = 0.0001 #improve initial training
        self.min_lr_ratio = 0.03 #I reduced it from 0.05 to 0.03 to help the model fine tune parameters, no signs of overfitting observed
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 16.0 # was set to 64 but batch size is 16

        #from looking at pictures I assume i need to adjust nms and confidence
        self.test_conf = 0.04
        self.nmsthre = 0.25
        # modify 'silu' to 'relu' for deployment on DPU
        self.act = 'relu'

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models.yolox_q import YOLOX
            from yolox.models.yolo_pafpn_deploy_q import YOLOPAFPN
            from yolox.models.yolo_head_q import YOLOXHead
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

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators.coco_evaluator_q import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator