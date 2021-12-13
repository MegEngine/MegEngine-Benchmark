# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 Megvii Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------
import argparse
import os
import time

import torch
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

import model as vit
import swin_transformer as swin_vit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        default="swin_tiny_patch4_window7_224",
        help="model architecture (default: swin_tiny_patch4_window7_224)",
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
        default=16,
        type=int,
        help="batch size for single GPU (default: 16)",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        default=0.01,
        help="learning rate for single GPU (default: 0.01)",
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
        choices=["normal", "mp"],
        help="Quantization Mode\n"
        "normal: no quantization, using float32\n"
        "mp: input type of conv and bn is fp16",
    )
    args = parser.parse_args()

    ngpus = args.ngpus
    if ngpus == 1:
        worker(0, 1, args)
    else:
        args.lr *= ngpus
        mp.spawn(worker, args=(ngpus, args), nprocs=ngpus)


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.shape) > 1:
            group_weight_decay.append(p)
        else:
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups


def worker(rank, world_size, args):
    cudnn.benchmark = True
    cudnn.enabled = True

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456',
                                world_size=world_size, rank=rank)

    if args.arch in vit.__dict__.keys():
        model = vit.__dict__[args.arch]()
    elif args.arch in swin_vit.__dict__.keys():
        model = swin_vit.__dict__[args.arch]()
    else:
        raise NotImplementedError

    if world_size > 1:
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        torch.cuda.set_device(rank)
        model.cuda(rank)

    optimizer = optim.SGD(
        get_parameters(model),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss().cuda(rank)
    def train_func(image, label):
        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        return loss

    # Start training
    objs = AverageMeter("Loss")
    total_time = AverageMeter("Time")

    image = np.random.randn(args.batch_size, 3, 224, 224).astype("float32")
    label = np.random.randint(0, 1000, size=(args.batch_size,))

    model.train()

    # warm up
    for step in range(10):
        n = args.batch_size
        _image = torch.from_numpy(image).cuda(rank)
        _label = torch.from_numpy(label).cuda(rank)
        with torch.cuda.amp.autocast(enabled=args.mode == "mp"):
            loss = train_func(_image, _label)
        optimizer.step()
        optimizer.zero_grad()
        loss.cpu().detach().numpy()

    for step in range(0, args.steps):
        n = args.batch_size
        t = time.time()
        _image = torch.from_numpy(image).cuda(rank)
        _label = torch.from_numpy(label).cuda(rank)
        with torch.cuda.amp.autocast(enabled=args.mode == "mp"):
            loss = train_func(_image, _label)
        optimizer.step()
        optimizer.zero_grad()
        objs.update(loss.cpu().detach().numpy(), n)
        # objs.update(0, n)

        total_time.update(time.time() - t)
        if step % args.print_freq == 0 and rank == 0:
            print(
                "TRAIN {} {:.4f} {} {}".format(
                    step,
                    args.lr,
                    objs,
                    total_time,
                )
            )
            objs.reset()

    if rank == 0:
        print("="*20, "summary", "="*20)
        print(" benchmark: vision_transformer(pytorch v{})".format(torch.__version__))
        print("      arch: {}".format(args.arch))
        print("train_mode: {}".format(args.mode))
        print(" batchsize: {}".format(args.batch_size))
        print("      #GPU: {}".format(args.ngpus))
        print("  avg time: {:.3f} seconds".format(total_time.avg))

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
