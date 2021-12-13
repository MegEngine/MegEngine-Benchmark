# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .atss_res50_coco_1x_800size import atss_res50_coco_1x_800size
from .atss_res101_coco_2x_800size import atss_res101_coco_2x_800size
from .atss_resx101_coco_2x_800size import atss_resx101_coco_2x_800size
from .faster_rcnn_res50_coco_1x_800size import faster_rcnn_res50_coco_1x_800size
from .faster_rcnn_res101_coco_2x_800size import faster_rcnn_res101_coco_2x_800size
from .faster_rcnn_resx101_coco_2x_800size import faster_rcnn_resx101_coco_2x_800size
from .fcos_res50_coco_1x_800size import fcos_res50_coco_1x_800size
from .fcos_res101_coco_2x_800size import fcos_res101_coco_2x_800size
from .fcos_resx101_coco_2x_800size import fcos_resx101_coco_2x_800size
from .retinanet_res50_coco_1x_800size import retinanet_res50_coco_1x_800size
from .retinanet_res101_coco_2x_800size import retinanet_res101_coco_2x_800size
from .retinanet_resx101_coco_2x_800size import retinanet_resx101_coco_2x_800size

_EXCLUDE = {}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
