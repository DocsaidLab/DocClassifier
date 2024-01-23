from enum import Enum
from typing import Union

import docsaidkit as D
import numpy as np

from .margin_based import Inference as MarginBasedInference

__all__ = [
    'DocClassifier', 'MarginBasedInference', 'ModelType'
]


class ModelType(D.EnumCheckMixin, Enum):
    margin_based = 1


class DocClassifier:

    def __init__(
        self,
        model_type: ModelType = ModelType.margin_based,
        model_cfg: str = None,
        backend: D.Backend = D.Backend.cpu,
        gpu_id: int = 0,
        threshold: float = None,
        register_root: Union[str, D.Path] = None,
        **kwargs
    ):
        model_type = ModelType.obj_to_enum(model_type)
        if model_type == ModelType.margin_based:
            model_cfg = 'lcnet050_arcface' if model_cfg is None else model_cfg
            valid_model_cfgs = list(MarginBasedInference.configs.keys())
            if model_cfg not in valid_model_cfgs:
                raise ValueError(
                    f'Invalid model_cfg: {model_cfg}, '
                    f'valid model_cfgs: {valid_model_cfgs}'
                )
            self.classifier = MarginBasedInference(
                gpu_id=gpu_id,
                backend=backend,
                model_cfg=model_cfg,
                threshold=threshold,
                register_root=register_root,
                **kwargs
            )

    def list_models(self) -> list:
        return list(self.classifier.configs.keys())

    def __call__(self, img: np.ndarray,) -> Union[str, None]:
        most_similar, max_score = self.classifier(img)
        return most_similar, max_score

    def __repr__(self) -> str:
        return f'{self.classifier.__class__.__name__}({self.classifier.model})'
