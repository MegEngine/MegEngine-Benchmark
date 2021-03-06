# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

import sys
sys.path.append("..")
import models

class CustomATSSConfig(models.ATSSConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnext101_32x8d"

        # ------------------------ training cfg ---------------------- #
        self.max_epoch = 36
        self.lr_decay_stages = [24, 32]


def atss_resx101_coco_2x_800size(**kwargs):
    r"""
    ATSS trained from COCO dataset.
    `"ATSS" <https://arxiv.org/abs/1912.02424>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = CustomATSSConfig()
    cfg.backbone_pretrained = False
    return models.ATSS(cfg, **kwargs)


Net = models.ATSS
Cfg = CustomATSSConfig
