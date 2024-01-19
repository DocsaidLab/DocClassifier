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
        **kwargs
    ):
        super().__init__()
        self.embed_features = nn.Sequential(
            GAP(),
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return normalize(self.embed_features(xs[-1]))
