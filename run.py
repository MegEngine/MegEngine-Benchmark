#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os
import argparse

parser = argparse.ArgumentParser(description='run benchmarks')
parser.add_argument('-b', '--benchmark', type=str, help='benchmark name', required=True)
parser.add_argument("-m","--mode",type=str,choices=["normal", "mp", "qat"],default="normal",
    help="Quantization Mode\n"
    "normal: no quantization, using float32\n"
    "mp: input type is fp16\n"
    "qat: quantization aware training",
)
parser.add_argument('-n', '--ngpus', type=int, help='number gpu', required=True)
parser.add_argument("--trace", action='store_true', default=False, help="whether use trace or not")
parser.add_argument("--symbolic", action='store_true', default=False, help="whether use symbolic trace or not")
parser.add_argument('--batch-size', type=int, help='batch size')
parser.add_argument('--steps', type=int, help='steps')
parser.add_argument('--loader', action="store_true", default=False, help="whether use dataloader")
parser.add_argument('--preload', action="store_true", default=False, help="whether use preloader")
args = parser.parse_args()

assert args.benchmark
assert args.ngpus
assert args.benchmark in [
    "shufflenet",
    "resnet",
    "faster_rcnn",
    "atss",
    "retinanet",
    "vision_transformer",
    "torch_resnet",
    "torch_shufflenet",
    "torch_vision_transformer",
] or args.benchmark.startswith("basecls_") or args.benchmark.startswith("timm_")

cmd = "source prepare.sh && "

if args.benchmark.startswith("torch_"):
    cmd += f"python3 ./pytorch/{args.benchmark[6:]}/train_random.py -n {args.ngpus}"
elif args.benchmark.startswith("basecls_"):
    cmd += f"python3 ./basecls/train_random.py -a {args.benchmark[8:]} -n {args.ngpus}"
elif args.benchmark.startswith("timm_"):
    cmd += f"python3 ./timm/train_random.py -a {args.benchmark[5:]} -n {args.ngpus}"
elif args.benchmark in ["faster_rcnn", "atss", "retinanet"]:
    cmd += f"python3 ./detection/train_random.py -a {args.benchmark} -n {args.ngpus}"
else:
    cmd += f"python3 ./{args.benchmark}/train_random.py -n {args.ngpus}"

if args.mode != "normal":
    cmd += f" -m {args.mode}"
if args.batch_size:
    cmd += f" -b {args.batch_size}"
if args.steps:
    cmd += f" -s {args.steps}"
if args.trace:
    cmd += " --trace"
if args.symbolic:
    cmd += " --symbolic"
if args.loader:
    cmd += " --loader"
if args.preload:
    cmd += " --preload"

print("command: ", cmd)
os.system('bash -c "{}"'.format(cmd))
