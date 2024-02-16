import random
from typing import Any, Callable, Tuple, Union

import albumentations as A
import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import numpy as np
from docsaidkit import INTER, Path

DIR = D.get_curdir(__file__)

INDOOR_ROOT = '/data/Dataset/indoor_scene_recognition/Images'

IMAGENET_ROOT = '/data/Dataset/ILSVRC2012/train'


class CoarseDropout(DT.BorderValueMixin, A.CoarseDropout):
    ...


class DefaultImageAug:

    def __init__(self, p=0.5):

        self.aug = A.Compose([

            A.OneOf([
                CoarseDropout(max_height=16, max_width=16,
                              min_holes=1, max_holes=4),
                A.RandomSunFlare(src_radius=128),
            ]),

            DT.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
            ),

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

    def __init__(self, image_size, p=0.5):
        h, w = image_size
        self.aug = A.Compose([

            A.OneOf([
                A.RandomResizedCrop(height=h, width=w),
                A.ColorJitter(),
            ])

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
        aug_func: Callable = None,
        aug_ratio: float = 0.0,
        length_of_dataset: int = 200000,
        return_tensor: bool = True,
        use_imagenet: bool = False,
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
        if use_imagenet:
            self.aug_func = ImageNetAug(image_size=image_size, p=aug_ratio)
        else:
            self.aug_func = aug_func(p=aug_ratio) if aug_func is not None \
                else DefaultImageAug(p=aug_ratio)
        self.dataset = self._build_dataset()

    def __len__(self) -> int:
        return self.length_of_dataset

    def _build_dataset(self):

        if self.use_imagenet:
            data_root = IMAGENET_ROOT
            cache_name = 'imagenet_cache.json'
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
        for label, f in enumerate(D.Tqdm(fs + fs_ind_, desc='Build Dataset')):

            if self.use_imagenet:
                dataset.append((label, str(f)))
            else:
                img = D.imresize(
                    img=D.imread(f),
                    size=self.image_size,
                    interpolation=self.interpolation
                )

                d01 = (label, img)
                d02 = (label * 24 + 1, D.imrotate(img, 90))
                d03 = (label * 24 + 2, D.imrotate(img, 180))
                d04 = (label * 24 + 3, D.imrotate(img, 270))
                d05 = (label * 24 + 4, cv2.flip(img, 0))
                d06 = (label * 24 + 5, cv2.flip(D.imrotate(img, 90), 0))
                d07 = (label * 24 + 6, cv2.flip(D.imrotate(img, 180), 0))
                d08 = (label * 24 + 7, cv2.flip(D.imrotate(img, 270), 0))
                d09 = (label * 24 + 8, d01[1][:img.shape[0] // 2, :, :])
                d10 = (label * 24 + 9, d02[1][:img.shape[0] // 2, :, :])
                d11 = (label * 24 + 10, d03[1][:img.shape[0] // 2, :, :])
                d12 = (label * 24 + 11, d04[1][:img.shape[0] // 2, :, :])
                d13 = (label * 24 + 12, d05[1][:img.shape[0] // 2, :, :])
                d14 = (label * 24 + 13, d06[1][:img.shape[0] // 2, :, :])
                d15 = (label * 24 + 14, d07[1][:img.shape[0] // 2, :, :])
                d16 = (label * 24 + 15, d08[1][:img.shape[0] // 2, :, :])
                d17 = (label * 24 + 16, d01[1][img.shape[0] // 2:, :, :])
                d18 = (label * 24 + 17, d02[1][img.shape[0] // 2:, :, :])
                d19 = (label * 24 + 18, d03[1][img.shape[0] // 2:, :, :])
                d20 = (label * 24 + 19, d04[1][img.shape[0] // 2:, :, :])
                d21 = (label * 24 + 20, d05[1][img.shape[0] // 2:, :, :])
                d22 = (label * 24 + 21, d06[1][img.shape[0] // 2:, :, :])
                d23 = (label * 24 + 22, d07[1][img.shape[0] // 2:, :, :])
                d24 = (label * 24 + 23, d08[1][img.shape[0] // 2:, :, :])

                dataset.extend([
                    d01, d02, d03, d04, d05, d06, d07, d08,
                    d09, d10, d11, d12, d13, d14, d15, d16,
                    d17, d18, d19, d20, d21, d22, d23, d24
                ])

        return dataset

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.dataset))
        label, base_img = self.dataset[idx]

        if self.use_imagenet:
            img = D.imread(base_img)
        else:
            img = base_img.copy()

        img = D.imresize(img, self.image_size, self.interpolation)

        img = self.aug_func(img)
        label = int(label)

        if self.return_tensor:
            img = img.transpose(2, 0, 1).astype('float32') / 255.0

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
