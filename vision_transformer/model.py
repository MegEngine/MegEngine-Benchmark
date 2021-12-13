# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
import warnings
from collections import OrderedDict
from typing import Callable, Optional, Union

import cv2
import megengine
import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

#========== backward compatibility ==========
try:
    gelu = F.gelu
except AttributeError:
    def gelu(x):
        return 0.5 * x * (1 + F.elemwise._elwise(x / math.sqrt(2), mode="erf"))

major, minor, *_ = megengine.__version__.split('.')
if int(major) <= 1 and int(minor) <= 4:
    LayerNorm = M.LayerNorm
else:
    class LayerNorm(M.Module):
        def __init__(self, normalized_shape, eps=1e-05, affine=True, **kwargs):
            super().__init__(**kwargs)
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.weight = megengine.Parameter(np.ones(self.normalized_shape, dtype="float32"))
                self.bias = megengine.Parameter(np.zeros(self.normalized_shape, dtype="float32"))
            else:
                self.weight = None
                self.bias = None

            self.reset_parameters()

        def reset_parameters(self):
            if self.affine:
                megengine.init.ones_(self.weight)
                megengine.init.zeros_(self.bias)

        def forward(self, x):
            x_shape = x.shape
            dim_delta = len(x_shape) - len(self.normalized_shape)
            non_flatten_shape = x_shape[:dim_delta]
            x = x.reshape(*non_flatten_shape, -1)

            mean = x.mean(axis=-1, keepdims=True)
            var = (x ** 2).mean(axis=-1, keepdims=True) - mean * mean

            x = (x - mean) / F.sqrt(var + self.eps)
            x = x.reshape(x_shape)
            if self.affine:
                x = self.weight * x + self.bias
            return x

        def _module_info_string(self) -> str:
            s = "normalized_shape={normalized_shape}, eps={eps}, affine={affine}"
            return s.format(**self.__dict__)


class DropPath(M.Dropout):
    def forward(self, x: mge.Tensor):
        if self.training:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = F.ones(shape)
            mask = F.dropout(mask, self.drop_prob, training=True)
            return x * mask
        else:
            return x


class PatchEmbed(M.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[M.Module] = None,
        flatten: bool = True,
        **kwargs,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = M.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else M.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        if self.flatten:
            x = F.flatten(x, 2).transpose(0, 2, 1)
        x = self.norm(x)
        return x


class Attention(M.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = M.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = M.Softmax(axis=-1)
        self.attn_drop = M.Dropout(attn_drop)
        self.proj = M.Linear(dim, dim)
        self.proj_drop = M.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = F.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(M.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_name: str = "gelu",
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = M.Linear(in_features, hidden_features)
        self.act = gelu
        self.fc2 = M.Linear(hidden_features, out_features)
        self.drop = M.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(M.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: M.Module = LayerNorm,
        act_name: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else M.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_name=act_name, drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(M.Module):
    """ViT model.

    Args:
        img_size: Input image size. Default: ``224``
        patch_size: Patch token size. Default: ``16``
        in_chans: Number of input image channels. Default: ``3``
        embed_dim: Number of linear projection output channels. Default: ``768``
        depth: Depth of Transformer Encoder layer. Default: ``12``
        num_heads: Number of attention heads. Default: ``12``
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: ``4.0``
        qkv_bias: If True, add a learnable bias to query, key, value. Default: ``True``
        drop_rate: Dropout rate. Default: ``0.0``
        attn_drop_rate: Attention dropout rate. Default: ``0.0``
        drop_path_rate: Stochastic depth rate. Default: ``0.0``
        embed_layer: Patch embedding layer. Default: ``PatchEmbed``
        norm_layer: Normalization layer. Default: ``LayerNorm``
        act_name: Activation function. Default: ``"gelu"``
        num_classes: Number of classes. Default: ``1000``
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: M.Module = PatchEmbed,
        norm_layer: M.Module = LayerNorm,
        act_name: str = "gelu",
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        # Patch Embedding
        self.embed_dim = embed_dim
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        # CLS & DST Tokens
        self.cls_token = megengine.Parameter(F.zeros([1, 1, embed_dim]))
        self.dist_token = None
        self.num_tokens = 1
        # Pos Embedding
        self.pos_embed = megengine.Parameter(F.zeros([1, num_patches + self.num_tokens, embed_dim]))
        self.pos_drop = M.Dropout(drop_rate)
        # Blocks
        dpr = [
            x.item() for x in F.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = M.Sequential(
            *[
                EncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_name=act_name,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.pre_logits = M.Identity()

        # Classifier head(s)
        self.head = M.Linear(self.embed_dim, num_classes) if num_classes > 0 else M.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = F.broadcast_to(self.cls_token, (x.shape[0], 1, self.cls_token.shape[-1]))
        x = F.concat((cls_token, x), axis=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x


def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model

def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model
