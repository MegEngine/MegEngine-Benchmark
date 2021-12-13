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
from loguru import logger
import megengine as mge
import megengine.distributed as dist
from basecls.utils import registers, set_nccl_env, set_num_threads
import benchmark

class X: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet50",
        help="model architecture (default: resnet50)",
    )
    parser.add_argument(
        "-n",
        "--ngpus",
        default=1,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=64,
        type=int,
        help="batch size for single GPU (default: 64)",
    )
    parser.add_argument(
        "--trace",
        action='store_true',
        default=False,
        help="whether use trace or not (default: False)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        default=200,
        type=int,
        help="number of train steps (default: 200)",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        default=0.025,
        help="learning rate for single GPU (default: 0.025)",
    )
    parser.add_argument("--momentum", default=0.9, help="momentum (default: 0.9)")
    parser.add_argument(
        "--weight-decay", default=1e-4, help="weight decay (default: 0.9)"
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
    new_args = X()
    new_args.model = args.arch
    new_args.mode = "train"
    new_args.device = "gpu"
    new_args.amp = 0 if args.mode == "normal" else 2
    new_args.fastrun = False
    new_args.trace = args.trace
    new_args.batch_size = args.batch_size
    new_args.channel = 3
    new_args.height = 224
    new_args.width = 224
    new_args.world_size = args.ngpus
    new_args.warm_iters = 50
    new_args.total_iters = args.steps
    new_args.log_seconds = 2

    set_nccl_env()
    set_num_threads()

    if new_args.world_size == 1:
        worker(new_args)
    else:
        dist.launcher(worker, n_gpus=new_args.world_size)(new_args)


def worker(args):
    rank = dist.get_rank()
    if rank != 0:
        logger.remove()
    logger.info(f"args: {args}")

    if args.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    mge.set_default_device(f"{args.device}{dist.get_rank()}")

    model = registers.models.get(args.model)(head=dict(w_out=1000))

    dataloader = benchmark.FakeDataLoader(
        args.batch_size,
        (args.height, args.width),
        args.channel,
        length=args.warm_iters + args.total_iters,
        num_classes=1000,
    )

    if args.mode == "train":
        BenchCls = benchmark.TrainBench
    elif args.mode == "eval":
        BenchCls = benchmark.EvalBench
    else:
        raise NotImplementedError(f"Benchmark mode '{args.mode}' not supported")

    bench = BenchCls(model, dataloader, args.trace, args.amp)
    avg_time = bench.benchmark(args.warm_iters, args.log_seconds)

    if rank == 0:
        print("="*20, "summary", "="*20)
        print(" benchmark: basecls")
        print("      arch: {}".format(args.model))
        print("train_mode: {}".format("normal" if args.amp == 0 else "mp"))
        print(" batchsize: {}".format(args.batch_size))
        print("      #GPU: {}".format(args.world_size))
        print("  avg time: {:.3f} seconds".format(avg_time/1000))

if __name__ == "__main__":
    main()
