#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
if os.environ["W_QUANT"]=='1':
    from pytorch_nndct.apis import torch_quantizer

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )

    parser.add_argument("--quant_mode", default='calib', type=str, help="mode for quantization")
    parser.add_argument("--quant_dir", default='quantized', type=str, help="directory for quantization")
    parser.add_argument("--is_dump", default=False, action="store_true", help="flag to dump xmodel")
    parser.add_argument("--fast_finetune", default=False, action="store_true", help="fast finetune for quantization")

    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = False

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="quant_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    device = torch.device('cuda')
    if args.is_dump:
        device = torch.device("cpu")
        args.batch_size = 1

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    # torch.cuda.set_device(rank)
    # model.cuda(rank)
    model.to(device)
    model.eval()

    # if not args.speed and not args.trt:
    if not args.speed:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        if 'model' in ckpt.keys():
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        logger.info("loaded checkpoint done.")

    import copy
    float_model = copy.deepcopy(model)
    if os.environ["W_QUANT"]=='1':
        dummy_input = torch.randn([1, 3, *exp.test_size]).to(device)
        quantizer = torch_quantizer(args.quant_mode, model, dummy_input, output_dir=args.quant_dir, device=device)
        model = quantizer.quant_model
        model.eval()
        if args.fast_finetune:
            if args.quant_mode == 'calib':
                sample_num = 2000
                fft_batch_size = 50
                quantizer.fast_finetune(feed_model_with_data, (model, evaluator.dataloader.dataset, device, sample_num, fft_batch_size))
            elif args.quant_mode == 'test':
                quantizer.load_ft_param()

    # if is_distributed:
    #     model = DDP(model, device_ids=[rank])

    # if args.fuse:
    #     logger.info("\tFusing model...")
    #     model = fuse_model(model)

    # if args.trt:
    #     assert (
    #         not args.fuse and not is_distributed and args.batch_size == 1
    #     ), "TensorRT model is not support model fusing and distributed inferencing!"
    #     trt_file = os.path.join(file_name, "model_trt.pth")
    #     assert os.path.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    # else:
    #     trt_file = None
    #     decoder = None
    trt_file = None
    decoder = None

    # start evaluate
    # *_, summary = evaluator.evaluate(
    #     model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    # )
    *_, summary = evaluator.evaluate(model, float_model, is_distributed, args.fp16, trt_file, decoder, exp.test_size,
                                     args.is_dump, device)
    if summary is not None:
        logger.info("\n" + summary)

    if os.environ["W_QUANT"]=='1':
        if args.quant_mode == 'calib':
            quantizer.export_quant_config()
        elif args.quant_mode == 'test' and args.is_dump:
            quantizer.export_xmodel(output_dir=args.quant_dir, deploy_check=True)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
