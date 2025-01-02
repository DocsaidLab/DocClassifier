from typing import Tuple, Union

import capybara as cb
import numpy as np

DIR = cb.get_curdir(__file__)

__all__ = ['Inference']


def preprocess(
    img: np.ndarray,
    img_size_infer: Tuple[int, int] = None,
    return_tensor: bool = True,
):
    if not cb.is_numpy_img(img):
        raise ValueError("Input image must be numpy array.")

    if img_size_infer is not None:
        img = cb.imresize(img, size=img_size_infer)

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
        '20240326': {
            'model_path': 'lcnet050_cosface_f256_r128_squeeze_lbn_imagenet_finetune_20240326_fp32.onnx',
            'file_id': '1sSv4wfmz-W5fdOtTA7oBKvdcv8rI2c0C',
            'img_size_infer': (128, 128),
            'threshold': 0.627  # FPR=0.01
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: cb.Backend = cb.Backencb.cpu,
        model_cfg: str = '20240326',
        threshold: float = None,
        register_root: Union[str, cb.Path] = None,
        **kwargs
    ):
        self.root = DIR / 'ckpt'
        self.cfg = cfg = self.configs[model_cfg]
        self.img_size_infer = cfg['img_size_infer']
        model_path = self.root / cfg['model_path']
        if not cb.Path(model_path).exists():
            cb.download_from_google(
                cfg['file_id'], model_path.name, str(DIR / 'ckpt'))
        self.model = cb.ONNXEngine(model_path, gpu_id, backend, **kwargs)
        self.bank = self.get_register(register_root)
        self.threshold = threshold if threshold is not None else cfg['threshold']

    @staticmethod
    def compare(feat1, feat2):
        return (np.dot(feat1, feat2) + 1) / 2

    def get_register(self, root: Union[str, cb.Path] = None):
        root = DIR.parent / 'register' if root is None else cb.Path(root)
        register = {}
        for f in cb.get_files(root, suffix=['.jpg', '.jpeg', '.png']):
            parts = f.relative_to(root).parts
            label = '_'.join(parts[:-1]) + '_' + f.stem
            if '_' in label[0]:
                label = label[1:]
            register[label] = self.extract_feature(cb.imread(f))
        return register

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
