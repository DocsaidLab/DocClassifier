import math
from typing import List, Tuple

import docsaidkit.torch as DT
import torch
import torch.nn as nn
from docsaidkit.torch import (GAP, Hsigmoid, build_backbone, build_transformer,
                              list_transformer, replace_module)
from torch.nn.functional import normalize


class Backbone(nn.Module):

    def __init__(self, name, replace_components: bool = False, **kwargs):
        super().__init__()
        self.backbone = build_transformer(name=name, **kwargs) \
            if name in list_transformer() else build_backbone(name=name, **kwargs)

        with torch.no_grad():
            dummy = torch.rand(1, 3, 128, 128)
            self.channels = [i.size(1) for i in self.backbone(dummy)]

        # For quantization
        if replace_components:
            replace_module(self.backbone, nn.Hardswish, nn.ReLU())
            replace_module(self.backbone, nn.Hardsigmoid, Hsigmoid())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


class FeatureLearningHead(nn.Module):

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        feature_map_size: int = None,
        **kwargs
    ):
        super().__init__()

        if feature_map_size is None:
            raise ValueError('image_size must be specified')

        in_dim = in_dim * feature_map_size * feature_map_size
        self.embed_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return normalize(self.embed_features(xs[-1]))


class FeatureLearningLNHead(nn.Module):

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        feature_map_size: int = None,
        **kwargs
    ):
        super().__init__()

        if feature_map_size is None:
            raise ValueError('image_size must be specified')

        in_dim_ = in_dim * feature_map_size * feature_map_size
        self.embed_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim_, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return normalize(self.embed_features(xs[-1]))


class FeatureLearningGAPHead(nn.Module):

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        **kwargs
    ):
        super().__init__()
        self.embed_features = nn.Sequential(
            GAP(),
            nn.Linear(in_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return normalize(self.embed_features(xs[-1]))


class FeatureLearningSqueezeHead(nn.Module):

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 128,
        feature_map_size: int = None,
        squeeze_ratio: float = 0.25,
        **kwargs
    ):
        super().__init__()

        if feature_map_size is None:
            raise ValueError('image_size must be specified')

        half_dim = int(in_dim * squeeze_ratio)
        in_dim_flatten = half_dim * feature_map_size * feature_map_size
        self.squeeze_feats = nn.Sequential(
            nn.Conv2d(in_dim, half_dim, 1, bias=False),
            nn.Flatten(),
        )

        self.embed_feats = nn.Sequential(
            nn.Linear(in_dim_flatten, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        feats = self.squeeze_feats(xs[-1])
        return normalize(self.embed_feats(feats))
