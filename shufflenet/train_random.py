# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import os
import time
import numpy as np
# pylint: disable=import-error
import model as snet_model
import quantizable_model as quantizable_snet_model

import megengine
import megengine.device as device
import megengine.autodiff as autodiff
import megengine.data as data
import megengine.data.transform as T
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optim
import megengine.jit as jit
import megengine.amp as amp
import megengine.quantization as Q

logging = megengine.logger.get_logger()

from dataset import get_dataloader

DEFAULT_QAT_CONFIG = {
    "ema":Q.ema_fakequant_qconfig,
    "ema_lowbi":Q.ema_lowbit_fakequant_qconfig,
    "sync_ema":Q.sync_ema_fakequant_qconfig,
    "min_max":Q.min_max_fakequant_qconfig,
    "tqt":Q.tqt_qconfig
}

def get_qconifg(config_name: str):
    return DEFAULT_QAT_CONFIG[config_name]


def main():
    parser = argparse.ArgumentParser(description="shufflenet benchmark")
    parser.add_argument(
        "-a",
        "--arch",
        default="shufflenet_v2_x2_0",
        help="model architecture (default: shufflenet_v2_x1_0)",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=1,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        default=200,
        type=int,
        help="number of train steps (default: 200)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=64,
        type=int,
        help="batch size for single GPU (default: 128)",
    )
    parser.add_argument(
        "--trace",
        action='store_true',
        default=False,
        help="whether use trace or not (default: False)",
    )
    parser.add_argument(
        "--symbolic",
        action='store_true',
        default=False,
        help="whether use symbolic trace or not (default: False)",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        default=0.001,
        help="learning rate for single GPU (default: 0.001)",
    )
    parser.add_argument("--momentum", default=0.9, help="momentum (default: 0.9)")
    parser.add_argument(
        "--weight-decay", default=4e-5, help="weight decay (default: 4e-5)"
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,
        metavar="N",
        help="print frequency (default: 1)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="normal",
        type=str,
        choices=["normal", "mp", "qat"],
        help="Quantization Mode\n"
        "normal: no quantization, using float32\n"
        "mp: input type is fp16\n"
        "qat: quantization aware training"
    )
    parser.add_argument(
        "--qat-config",
        default="min_max",
        type=str,
        choices=["min_max", "ema", "ema_lowbit", "sync_ema", "tqt"],
        help="quantization aware training config\n"
        "min_max: min_max_fakequant_qconfig\n"
        "ema: ema_fakequant_qconfig\n"
        "ema_lowbit: ema_lowbit_fakequant_qconfig\n"
        "sync_ema: sync_ema_fakequant_qconfig\n"
        "tqt: tqt_qconfig"
    )

    parser.add_argument("--dist-addr", default="localhost")
    parser.add_argument("--dist-port", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--rank", default=0)
    parser.add_argument("--loader", default=False, action="store_true", help="whether use loader")
    parser.add_argument("--preload", default=False, action="store_true", help="whether use preload")

    args = parser.parse_args()

    if args.world_size is None:
        args.world_size = args.ngpus
    if args.world_size > 1:
        # launch processes
        train_func = dist.launcher(worker, master_ip=args.dist_addr, port=args.dist_port,
                                world_size=args.world_size, n_gpus=args.ngpus, rank_start=args.rank * args.ngpus)
        train_func(args)
    else:
        worker(args)


def worker(args):
    steps = args.steps
    # build model
    shufflenet = quantizable_snet_model if args.mode == "qat" else snet_model
    model = shufflenet.__dict__[args.arch]()

    if args.mode == "qat":
        if args.qat_config == "sync_ema":
            assert args.ngpus > 1, "sync_ema does not support ngpus={}".format(args.ngpus)
        qconfig = get_qconifg(args.qat_config)
        model = Q.quantize_qat(module=model, qconfig= qconfig)
        model.train()
        Q.enable_observer(model)
        Q.enable_fake_quant(model)

    # Sync parameters
    if args.world_size > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=dist.make_allreduce_cb("SUM") if args.world_size > 1 else None,
    )

    # Optimizer
    params_wd = []
    params_nwd = []
    params_scale = []
    for n, p in model.named_parameters():
        if n.find("weight") >= 0 and len(p.shape) > 1:
            params_wd.append(p)
        elif n.find("scale") >= 0:
            params_scale.append(p)
        else:
            params_nwd.append(p)
    opt = optim.SGD(
        [{"params": params_wd},
         {"params": params_nwd, "weight_decay": 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay * args.world_size,  # scale weight decay in "SUM" mode
    )

    # train and valid func
    @amp.autocast(enabled=args.mode == "mp")
    def train_step(image, label):
        with gm:
            logits = model(image)
            loss = F.nn.cross_entropy(logits, label, label_smooth=0.1)
            gm.backward(loss)
        opt.step().clear_grad()
        return loss

    if args.trace:
        if args.symbolic:
            train_step = jit.trace(train_step, symbolic=True, sublinear_memory_config=jit.SublinearMemoryConfig(genetic_nr_iter=50), symbolic_shape=False)
        else:
            train_step = jit.trace(train_step, symbolic=False, symbolic_shape=False)
    else:
        assert args.symbolic==False, "invalid arguments: trace=Trace, symbolic=True"

    # start training
    objs = AverageMeter("Loss")
    clck = AverageMeter("Time")

    if args.loader:
        dataloader = iter(get_dataloader(args))
        image,label = next(dataloader)
    else:
        image = np.random.randn(args.batch_size, 3, 224, 224).astype("float32")
        label = np.random.randint(0, 1000, size=(args.batch_size,)).astype("int32")

    # warm up
    for step in range(10):

        if args.loader:
            image,label = next(dataloader)
            if not args.preload:
                image = megengine.tensor(image, dtype="float32")
                label = megengine.tensor(label, dtype="int32")
        else:
            image = megengine.tensor(image, dtype="float32")
            label = megengine.tensor(label, dtype="int32")

        loss = train_step(image, label)
        loss.item()

    for step in range(0, steps):
        t = time.time()
        if args.loader:
            image,label = next(dataloader)
            if not args.preload:
                image = megengine.tensor(image, dtype="float32")
                label = megengine.tensor(label, dtype="int32")
        else:
            image = megengine.tensor(image, dtype="float32")
            label = megengine.tensor(label, dtype="int32")

        loss = train_step(image, label)
        objs.update(loss.item())

        clck.update(time.time() - t)
        if step % args.print_freq == 0 and dist.get_rank() == 0:
            print(
                "Step {}, {}, {}".format(
                step,
                objs,
                clck,
            ))
            objs.reset()

    if dist.get_rank() == 0:
        print("="*20, "summary", "="*20)
        print(" benchmark: shufflent")
        if args.trace:
            print("      mode: trace(symbolic={})".format("True, sublinear=True" if args.symbolic else "False"))
        else:
            print("      mode: imperative")
        print("    loader: {}".format("" if not args.loader else "--loader"))
        if args.loader:
            print("   preload: {}".format("" if not args.preload else "--preload"))
        print("      arch: {}".format(args.arch))
        print("train_mode: {}".format(args.mode))
        print(" batchsize: {}".format(args.batch_size))
        print("      #GPU: {}".format(args.ngpus))
        print("  avg time: {:.3f} seconds".format(clck.avg))

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    main()
