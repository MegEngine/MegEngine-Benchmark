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

def fcos_res50_coco_1x_800size(**kwargs):
    r"""
    FCOS trained from COCO dataset.
    `"FCOS" <https://arxiv.org/abs/1904.01355>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = models.FCOSConfig()
    cfg.backbone_pretrained = False
    return models.FCOS(cfg, **kwargs)


Net = models.FCOS
Cfg = models.FCOSConfig
