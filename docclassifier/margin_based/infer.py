import os
from typing import Tuple, Union

import docsaidkit as D
import numpy as np

DIR = D.get_curdir(__file__)

__all__ = ['Inference']


def preprocess(
    img: np.ndarray,
    img_size_infer: Tuple[int, int] = None,
    return_tensor: bool = True,
):
    if not D.is_numpy_img(img):
        raise ValueError("Input image must be numpy array.")

    if img_size_infer is not None:
        img = D.imresize(img, size=img_size_infer)

    if return_tensor:
        img = np.transpose(img, axes=(2, 0, 1)).astype('float32')
        img = img[None] / 255.

    return {
        'input': {'img': img},
        'img_size_infer': img_size_infer,
        'return_tensor': return_tensor,
    }


class Inference:

    configs = {
        'lcnet050_cosface': {
            'model_path': 'lcnet050_cosface_f256_r128_fp32.onnx',
            'file_id': '1wTER45MAVuNS93nGu5x21jOEGkdp25Wn',
            'img_size_infer': (128, 128),
            'threshold': 0.663  # FPR=0.01
        },
        'lcnet050_cosface_squeeze': {
            'model_path': 'lcnet050_cosface_f256_r128_squeeze_fp32.onnx',
            'file_id': '1xDr0TuQlYfZsGgvSd0rFCCT6LMnQU1A4',
            'img_size_infer': (128, 128),
            'threshold': 0.645  # FPR=0.01
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: D.Backend = D.Backend.cpu,
        model_cfg: str = 'lcnet050_cosface',
        threshold: float = None,
        register_root: Union[str, D.Path] = None,
        **kwargs
    ):
        self.root = DIR / 'ckpt'
        self.cfg = cfg = self.configs[model_cfg]
        self.img_size_infer = cfg['img_size_infer']
        model_path = str(self.root / cfg['model_path'])
        if not D.Path(model_path).exists():
            file_id = cfg['file_id']
            os.system(D.gen_download_cmd(file_id, model_path))
        self.model = D.ONNXEngine(model_path, gpu_id, backend, **kwargs)
        self.bank = self.get_register(register_root)
        self.threshold = threshold if threshold is not None else cfg['threshold']

    @staticmethod
    def compare(feat1, feat2):
        return (np.dot(feat1, feat2) + 1) / 2

    def get_register(self, root: Union[str, D.Path] = None):
        root = DIR.parent / 'register' if root is None else D.Path(root)
        return {
            f.stem: self.extract_feature(D.imread(f))
            for f in D.get_files(root, suffix=['.jpg', '.jpeg', '.png'])
        }

    def extract_feature(self, img: np.ndarray) -> np.ndarray:
        img_infos = preprocess(img, img_size_infer=self.img_size_infer)
        feats = self.model(**img_infos['input'])
        return feats['feats'][0]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        feats = self.extract_feature(img)
        score_pool = []
        for k, v in self.bank.items():
            score = self.compare(feats, v)
            if score > self.threshold:
                score_pool.append((k, score))
        most_similar, max_score = max(
            score_pool, key=lambda x: x[1], default=(None, 0.))
        return most_similar, max_score
