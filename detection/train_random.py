# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import bisect
import os
import time
import pickle
import numpy as np
import megengine.amp as amp

import megengine as mge
import megengine.distributed as dist
from megengine.autodiff import GradManager
from megengine.data import DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from megengine.optimizer import SGD
import megengine.jit as jit

from tools.utils import (
    AverageMeter,
    DetectionPadCollator,
    GroupedRandomSampler,
    PseudoDetectionDataset,
    get_config_info,
    import_from_file
)
logger = mge.get_logger(__name__)
logger.setLevel("INFO")
mge.device.set_prealloc_config(1024, 1024, 512 * 1024 * 1024, 2.0)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--arch", type=str, help="model architecture",
    )
    parser.add_argument(
        "-n", "--ngpus", default=1, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="batchsize for training",
    )
    parser.add_argument(
        "-s", "--steps", default=100, type=int, help="number of train steps (default: 100)",
    )
    parser.add_argument(
        "--trace",
        action='store_true',
        default=False,
        help="whether use trace or not (default: False)",
    )
    parser.add_argument(
        "--preloader",
        action='store_true',
        default=False,
        help="whether use preloader or not (default: False)",
    )
    parser.add_argument(
        "--symbolic",
        action='store_true',
        default=False,
        help="whether use symbolic trace or not (default: False)",
    )
    parser.add_argument(
        "-d", "--loader", default=False, action="store_true", help="use pseudo detection dataset loader",
    )
    parser.add_argument(
        "-p", "--print-freq", default=1, type=int, help="print frequency (default: 1)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="normal",
        type=str,
        choices=["normal", "mp"],
        help="Quantization Mode\n"
        "normal: no quantization, using float32\n"
        "mp: input type is fp16",
    )
    parser.add_argument("--preload", default=False, action="store_true", help="whether use preload")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    # ------------------------ begin training -------------------------- #
    if args.ngpus > 1:
        train_func = dist.launcher(worker, n_gpus=args.ngpus)
        train_func(args)
    else:
        worker(args)


def worker(args):
    config_file = {
        "faster_rcnn": "configs/faster_rcnn_res50_coco_1x_800size.py",
        "atss": "configs/atss_res50_coco_1x_800size.py",
        "retinanet": "configs/retinanet_res50_coco_1x_800size.py",
    }[args.arch]

    current_network = import_from_file(config_file)

    model = current_network.Net(current_network.Cfg())

    model.train()

    # if dist.get_rank() == 0:
    #     logger.info(get_config_info(model.cfg))
    #     logger.info(repr(model))

    params_with_grad = []
    for name, param in model.named_parameters():
        if "bottom_up.conv1" in name and model.cfg.backbone_freeze_at >= 1:
            continue
        if "bottom_up.layer1" in name and model.cfg.backbone_freeze_at >= 2:
            continue
        params_with_grad.append(param)

    opt = SGD(
        params_with_grad,
        lr=model.cfg.basic_lr * args.batch_size,
        momentum=model.cfg.momentum,
        weight_decay=model.cfg.weight_decay * dist.get_world_size(),
    )

    gm = GradManager()
    if dist.get_world_size() > 1:
        gm.attach(
            params_with_grad,
            callbacks=[dist.make_allreduce_cb("SUM", dist.WORLD)]
        )
    else:
        gm.attach(params_with_grad)

    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)  # sync parameters

    train_loader = None

    for epoch in range(1):
        train_one_epoch(model, train_loader, opt, gm, epoch, args)

def train_one_epoch(model, data_queue, opt, gm, epoch, args):
    @amp.autocast(enabled=args.mode == "mp")
    def train_func(image, im_info, gt_boxes):
        with gm:
            loss_dict = model(image=image, im_info=im_info, gt_boxes=gt_boxes)
            gm.backward(loss_dict["total_loss"])
            loss_list = list(loss_dict.values())
        opt.step().clear_grad()
        return loss_list

    if args.trace:
        if args.symbolic:
            train_func = jit.trace(train_func, symbolic=True, sublinear_memory_config=jit.SublinearMemoryConfig(genetic_nr_iter=50), symbolic_shape=True)
        else:
            train_func = jit.trace(train_func, symbolic=False, symbolic_shape=False)
    else:
        assert args.symbolic==False, "invalid arguments: trace=Trace, symbolic=True"

    loss_meter = AverageMeter(record_len=model.cfg.num_losses)
    time_meter = AverageMeter(record_len=2)

    log_interval = model.cfg.log_interval
    tot_step = model.cfg.nr_images_epoch // (args.batch_size * dist.get_world_size())

    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_dir, 'batch.pkl') # batch_size for batch.pkl is 2
    mini_batch = pickle.load(open(data_path, "rb"))

    if args.batch_size != 2:
        repeats = (args.batch_size+1) // 2
        mini_batch['data'] = np.concatenate([mini_batch['data'],]*repeats)[:args.batch_size]
        mini_batch['im_info'] = np.concatenate([mini_batch['im_info'],]*repeats)[:args.batch_size]
        mini_batch['gt_boxes'] = np.concatenate([mini_batch['gt_boxes'],]*repeats)[:args.batch_size]

    # warm up
    for step in range(10):
        if data_queue:
            mini_batch = next(data_queue)
        loss_list = train_func(
            image=mge.tensor(mini_batch["data"]),
            im_info=mge.tensor(mini_batch["im_info"]),
            gt_boxes=mge.tensor(mini_batch["gt_boxes"])
        )
        _ = [loss.numpy() for loss in loss_list]

    for step in range(args.steps):
        tik = time.time()
        if data_queue:
            mini_batch = next(data_queue)
        data_tok = time.time()

        loss_list = train_func(
            image=mge.tensor(mini_batch["data"]),
            im_info=mge.tensor(mini_batch["im_info"]),
            gt_boxes=mge.tensor(mini_batch["gt_boxes"])
        )

        loss_meter.update([loss.numpy().item() for loss in loss_list])
        tok = time.time()

        time_meter.update([tok - tik, data_tok - tik])

        if step % args.print_freq == 0 and dist.get_rank() == 0:
            print(
                "Step {}, Loss ({}), Time (tot:{:.3f}, data:{:.3f})".format(
                step,
                "".join(["{:.3f} ".format(t) for t in loss_meter.average()]),
                *time_meter.average(),
            ))
            loss_meter.reset()

    if dist.get_rank() == 0:
        print("="*20, "summary", "="*20)
        print(" benchmark: detection")
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
        print("  avg time: {:.3f} seconds".format(time_meter.average()[0]))


# pylint: disable=unused-argument
def build_dataset(dataset_dir, cfg):
    return PseudoDetectionDataset(order=["image", "boxes", "boxes_category", "info"])


# pylint: disable=dangerous-default-value
def build_sampler(train_dataset, batch_size, aspect_grouping=[1]):
    def _compute_aspect_ratios(dataset):
        aspect_ratios = []
        for i in range(len(dataset)):
            info = dataset.get_img_info(i)
            aspect_ratios.append(info["height"] / info["width"])
        return aspect_ratios

    def _quantize(x, bins):
        return list(map(lambda y: bisect.bisect_right(sorted(bins), y), x))

    if len(aspect_grouping) == 0:
        return Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))

    aspect_ratios = _compute_aspect_ratios(train_dataset)
    group_ids = _quantize(aspect_ratios, aspect_grouping)
    return Infinite(GroupedRandomSampler(train_dataset, batch_size, group_ids))


def build_dataloader(batch_size, dataset_dir, cfg, preloader= False):
    train_dataset = build_dataset(dataset_dir, cfg)
    train_sampler = build_sampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.ShortestEdgeResize(
                    cfg.train_image_short_size,
                    cfg.train_image_max_size,
                    sample_style="choice",
                ),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ],
            order=["image", "boxes", "boxes_category"],
        ),
        collator=DetectionPadCollator(),
        num_workers=8,
        preload= preloader,
    )
    return train_dataloader


if __name__ == "__main__":
    main()
