import random
import warnings
from typing import Any, Callable, List, Tuple, Union

import albumentations as A
import clip
import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import numpy as np
import torch
from docsaidkit import INTER, Path
from PIL import Image
from torch.nn.functional import normalize

DIR = D.get_curdir(__file__)

INDOOR_ROOT = '/data/Dataset/indoor_scene_recognition/Images'

IMAGENET_ROOT = '/data/Dataset/ILSVRC2012/train'


class CoarseDropout(DT.BorderValueMixin, A.CoarseDropout):
    ...


class DefaultImageAug:

    def __init__(self, image_size=(128, 128), p=0.5):
        h, w = image_size
        self.aug = A.Compose([

            A.OneOf([
                A.RandomResizedCrop(height=h, width=w, scale=(0.8, 1.0)),
                DT.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                )
            ]),

            A.OneOf([
                A.MotionBlur(),
                A.MedianBlur(),
                A.GaussianBlur(),
                A.ZoomBlur(),
                A.Defocus(radius=(3, 5)),
                A.ImageCompression(quality_lower=0, quality_upper=50),
            ]),

            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.MultiplicativeNoise(
                    multiplier=[0.5, 1.5],
                    elementwise=True,
                    per_channel=True
                ),
            ]),

            A.OneOf([
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.1,
                    saturation=0.1,
                ),
                A.ToGray(),
                A.ToSepia(),
                A.ChannelShuffle(),
                A.ChannelDropout(),
                A.RGBShift(),
                A.InvertImg(),
            ]),

        ], p=p)

    def __call__(self, img: np.ndarray) -> Any:
        img = self.aug(image=img)['image']
        return img


class ImageNetAug:

    def __init__(self, image_size=(128, 128), p=0.5):
        h, w = image_size
        self.aug = A.Compose([

            A.RandomResizedCrop(height=h, width=w, scale=(0.8, 1.0)),
            DT.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
            )

        ], p=p)

    def __call__(self, img: np.ndarray) -> Any:
        img = self.aug(image=img)['image']
        return img


class SyncDataset:

    def __init__(
        self,
        root: Union[str, Path] = None,
        image_size: Tuple[int, int] = None,
        interpolation: Union[str, int, INTER] = INTER.BILINEAR,
        aug_ratio: float = 0.0,
        length_of_dataset: int = 200000,
        return_tensor: bool = True,
        use_imagenet: bool = False,
        use_clip: bool = False,
        **kwargs
    ) -> None:
        self.image_size = image_size
        self.interpolation = interpolation
        self.length_of_dataset = length_of_dataset
        self.return_tensor = return_tensor
        self.aug_ratio = aug_ratio
        self.root = DIR.parent / 'data' / 'unique_pool' \
            if root is None else Path(root)
        self.use_imagenet = use_imagenet
        self.use_clip = use_clip
        self.aug_func = ImageNetAug(image_size=image_size, p=aug_ratio) \
            if use_imagenet else DefaultImageAug(image_size=image_size, p=aug_ratio)

        if self.use_clip:
            self.clip_model, self.preprocess = clip.load(
                'ViT-B/32', device='cpu')

        self._build_dataset()

    def __len__(self) -> int:
        return self.length_of_dataset

    def _clip(self, file_path: Union[str, Path, np.ndarray]) -> List[float]:
        if self.use_clip:
            with torch.no_grad():
                if isinstance(file_path, np.ndarray):
                    tensor = Image.fromarray(file_path)
                else:
                    tensor = Image.open(str(file_path))
                tensor = self.preprocess(tensor).unsqueeze(0).to('cpu')
                clip_feats = self.clip_model.encode_image(tensor)
                clip_feats = normalize(clip_feats, dim=-1)[0]
                clip_feats = clip_feats.detach().cpu().numpy()
            return clip_feats.tolist()
        return False

    def _build_dataset(self):

        if self.use_imagenet:
            data_root = IMAGENET_ROOT
            cache_name = 'imagenet_cache.json'
            data_cache = 'imagenet_data_cache.json'
        else:
            data_root = INDOOR_ROOT
            cache_name = 'indoor_cache.json'

        if not (fp := DIR.parent / 'data' / cache_name).is_file():
            fs_ind = D.get_files(data_root, suffix=['.jpg', '.png', '.jpeg'])
            fs_ind_ = [str(f) for f in D.Tqdm(
                fs_ind, desc='Drop Empty images.') if D.imread(f) is not None]
            D.dump_json(fs_ind_, fp)
        else:
            fs_ind_ = D.load_json(fp)

        fs = D.get_files(self.root, suffix=['.jpg', '.png', '.jpeg'])

        dataset = []

        if self.use_imagenet:

            if not (fp_data := DIR.parent / 'data' / data_cache).is_file():
                for label, f in enumerate(D.Tqdm(fs + fs_ind_, desc='Build Dataset')):
                    dataset.append((label, str(f), self._clip(f)))
                D.dump_json(dataset, fp_data)
            else:
                dataset = D.load_json(fp_data)
        else:

            if self.use_clip:
                warnings.warn(
                    'Clip model is not available for indoor dataset.')

            for label, f in enumerate(D.Tqdm(fs + fs_ind_, desc='Build Dataset')):
                img = D.imresize(
                    img=D.imread(f),
                    size=self.image_size,
                    interpolation=self.interpolation
                )

                h_half = img.shape[0] // 2

                d01 = (label * 24 + 1, img, False)
                d02 = (label * 24 + 2, D.imrotate(img, 90), False)
                d03 = (label * 24 + 3, D.imrotate(img, 180), False)
                d04 = (label * 24 + 4, D.imrotate(img, 270), False)
                d05 = (label * 24 + 5, cv2.flip(img, 0), False)
                d06 = (label * 24 + 6, cv2.flip(d02[1], 0), False)
                d07 = (label * 24 + 7, cv2.flip(d03[1], 0), False)
                d08 = (label * 24 + 8, cv2.flip(d04[1], 0), False)
                d09 = (label * 24 + 9, d01[1][:h_half, :, :], False)
                d10 = (label * 24 + 10, d02[1][:h_half, :, :], False)
                d11 = (label * 24 + 11, d03[1][:h_half, :, :], False)
                d12 = (label * 24 + 12, d04[1][:h_half, :, :], False)
                d13 = (label * 24 + 13, d05[1][:h_half, :, :], False)
                d14 = (label * 24 + 14, d06[1][:h_half, :, :], False)
                d15 = (label * 24 + 15, d07[1][:h_half, :, :], False)
                d16 = (label * 24 + 16, d08[1][:h_half, :, :], False)
                d17 = (label * 24 + 17, d01[1][h_half:, :, :], False)
                d18 = (label * 24 + 18, d02[1][h_half:, :, :], False)
                d19 = (label * 24 + 19, d03[1][h_half:, :, :], False)
                d20 = (label * 24 + 20, d04[1][h_half:, :, :], False)
                d21 = (label * 24 + 21, d05[1][h_half:, :, :], False)
                d22 = (label * 24 + 22, d06[1][h_half:, :, :], False)
                d23 = (label * 24 + 23, d07[1][h_half:, :, :], False)
                d24 = (label * 24 + 24, d08[1][h_half:, :, :], False)

                dataset.extend([
                    d01, d02, d03, d04, d05, d06, d07, d08,
                    d09, d10, d11, d12, d13, d14, d15, d16,
                    d17, d18, d19, d20, d21, d22, d23, d24
                ])

        label, img, clip_feat = [], [], []
        for d in D.Tqdm(dataset):
            label.append(d[0])
            img.append(d[1])
            clip_feat.append(np.array(d[2], dtype='float32'))

        self.lbs = np.array(label, dtype='int64')
        if self.use_imagenet:
            self.imgs = np.stack(img, axis=0)
        else:
            self.imgs = img
        self.feats = np.array(clip_feat)

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.lbs))

        label = self.lbs[idx]
        img = self.imgs[idx]

        # Loading image from file, because imagenet dataset is too large
        if self.use_imagenet:
            img = D.imread(img)

        img = D.imresize(img, self.image_size, self.interpolation)

        if self.use_clip:
            clip_feat = self.feats[idx]
            base_img = img.copy()

        img = self.aug_func(img)
        label = int(label)

        if self.return_tensor:
            img = img.transpose(2, 0, 1).astype('float32') / 255.0

            if self.use_clip:
                base_img = base_img.transpose(2, 0, 1)
                base_img = base_img.astype('float32') / 255.0
                clip_feat = clip_feat.astype('float32')

        if self.use_clip:
            return img, label, base_img, clip_feat

        return img, label


class RealDataset:

    label_mapper = {
        'IDCardFront': 0,
        'IDCardBack': 1,
        'DriverLicenseFront': 2,
        'HealthIDCard': 3,
        'ResidentIDCardFront': 4,
        'ResidentIDCardBack': 5,
        'VehicleLicense': 6
    }

    def __init__(
        self,
        root: Union[str, Path] = None,
        image_size: Tuple[int, int] = None,
        interpolation: Union[str, int, INTER] = INTER.BILINEAR,
        return_tensor: bool = True,
        **kwargs
    ):
        self.image_size = image_size
        self.interpolation = interpolation
        self.return_tensor = return_tensor
        self.root = DIR.parent / 'benchmark' / 'docs_benchmark_dataset' \
            if root is None else Path(root)
        self.dataset = self._build_dataset()

    def __len__(self):
        return len(self.dataset)

    def _build_dataset(self):
        if (fp := DIR.parent / 'benchmark' / 'real_cache.json').is_file():
            ds = D.load_json(fp)
        else:
            fs = D.get_files(self.root, suffix=['.jpg', '.png', '.jpeg'])
            ds = [(f.parent.name, str(f))
                  for f in fs if f.parent.name != 'Passport']
            local_random = random.Random()
            local_random.seed(42)
            local_random.shuffle(ds)
            D.dump_json(ds, fp)
        return ds

    def __getitem__(self, idx):
        label, file = self.dataset[idx]
        img = D.imread(file)
        img = D.imresize(img, self.image_size)
        if self.return_tensor:
            img = img.transpose(2, 0, 1).astype('float32') / 255.0
        return img, self.label_mapper[label]
