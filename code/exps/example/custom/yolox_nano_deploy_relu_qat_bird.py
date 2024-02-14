#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from pytorch_nndct import QatProcessor

import os

import torch
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25

        #self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)

        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
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
        self.max_epoch = 50
        # modified for 2 training
        self.warmup_epochs = 10 #increase warmup to 10
        # minimum learning rate during warmup
        self.warmup_lr = 0.0001 #improve initial training
        self.min_lr_ratio = 0.03 #I reduced it from 0.05 to 0.03 to help the model fine tune parameters, no signs of overfitting observed
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 8.0 

        #from looking at pictures I assume i need to adjust nms and confidence
        self.test_conf = 0.04
        self.nmsthre = 0.25
        # modify 'silu' to 'relu' for deployment on DPU
        self.act = 'relu'

        # QAT
        self.is_qat = True
        self.float_ckpt = 'float/baseline.pth'
        self.calib_dir = 'quantize_result'
        self.thresh_lr_scale = 10

        self.device = torch.device('cuda')
        

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

        if self.is_qat:
            if self.float_ckpt:
                ckpt = torch.load(self.float_ckpt)
                print('Loading float model weight for QAT from {}'.format(self.float_ckpt))
                self.model.load_state_dict(ckpt["model"])
            self.model.to(self.device)
            dummy_input = torch.randn([1, 3, *self.test_size], dtype=torch.float32).to(self.device)
            self.qat_processor = QatProcessor(self.model, dummy_input, bitwidth=8, mix_bit=False, device=self.device)
            if os.path.exists(self.calib_dir):
                print('Loading calibration result for QAT initialization from {}'.format(self.calib_dir))
            else:
                self.calib_dir = ''
                print('The calib_dir: {} is not exists, we set it empty as default'.format(self.calib_dir))
            self.model = self.qat_processor.trainable_model(calib_dir=self.calib_dir)

        return self.model


    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if isinstance(v, nn.Identity): # during nndct's QAT, 'bn' is merged to 'conv' and replaced by Identity
                        continue
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            threshold = [ 
                param for name, param in self.model.named_parameters()
                if 'threshold' in name
            ]
            # print("Threshold params: ")
            # [print(name) for name, param in self.model.named_parameters() if 'threshold' in name]
            q_param_group = { 
                'params': threshold,
                'lr': lr * self.thresh_lr_scale,
                'name': 'threshold'
            } 

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group(q_param_group)
            self.optimizer = optimizer

        return self.optimizer


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

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        return evaluator.evaluate(model, model, is_distributed, half, return_outputs=return_outputs)
