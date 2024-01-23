from typing import Any, Callable, Tuple, Union

import albumentations as A
import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import numpy as np
from docsaidkit import INTER, Path

DIR = D.get_curdir(__file__)

INDOOR_ROOT = '/data/Dataset/indoor_scene_recognition/Images'


class DefaultImageAug:

    def __init__(self, p=0.5):

        self.aug = A.Compose([

            DT.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=10,
            ),

            A.OneOf([
                A.MotionBlur(),
                A.Defocus(radius=(3, 5)),
            ], p=p),

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
        **kwargs
    ) -> None:
        self.image_size = image_size
        self.interpolation = interpolation
        self.length_of_dataset = length_of_dataset
        self.return_tensor = return_tensor
        self.aug_ratio = aug_ratio
        self.root = DIR.parent / 'data' / 'unique_pool' \
            if root is None else Path(root)
        self.aug_func = aug_func(p=aug_ratio) if aug_func is not None \
            else DefaultImageAug(p=aug_ratio)
        self.dataset = self._build_dataset()

    def __len__(self) -> int:
        return self.length_of_dataset

    def _build_dataset(self):

        if not (fp := DIR.parent / 'data' / 'indoor_cache.json').is_file():
            fs_ind = D.get_files(INDOOR_ROOT, suffix=['.jpg', '.png', '.jpeg'])
            fs_ind_ = [str(f) for f in D.Tqdm(
                fs_ind, desc='Drop Empty images.') if D.imread(f) is not None]
            D.dump_json(fs_ind_, fp)
        else:
            fs_ind_ = D.load_json(fp)

        fs = D.get_files(self.root, suffix=['.jpg', '.png', '.jpeg'])

        dataset = []
        for label, f in enumerate(D.Tqdm(fs + fs_ind_, desc='Build Dataset')):

            img = D.imresize(
                img=D.imread(f),
                size=self.image_size,
                interpolation=self.interpolation
            )

            d01 = (label * 24, img)
            d02 = (label * 24 + 1, D.imrotate(img, 90))
            d03 = (label * 24 + 2, D.imrotate(img, 180))
            d04 = (label * 24 + 3, D.imrotate(img, 270))
            d05 = (label * 24 + 4, cv2.flip(img, 0))
            d06 = (label * 24 + 5, cv2.flip(D.imrotate(img, 90), 0))
            d07 = (label * 24 + 6, cv2.flip(D.imrotate(img, 180), 0))
            d08 = (label * 24 + 7, cv2.flip(D.imrotate(img, 270), 0))
            d09 = (label * 24 + 8, cv2.flip(img, 1))
            d10 = (label * 24 + 9, cv2.flip(D.imrotate(img, 90), 1))
            d11 = (label * 24 + 10, cv2.flip(D.imrotate(img, 180), 1))
            d12 = (label * 24 + 11, cv2.flip(D.imrotate(img, 270), 1))
            d13 = (label * 24 + 12, 255 - d01[1])
            d14 = (label * 24 + 13, 255 - d02[1])
            d15 = (label * 24 + 14, 255 - d03[1])
            d16 = (label * 24 + 15, 255 - d04[1])
            d17 = (label * 24 + 16, 255 - d05[1])
            d18 = (label * 24 + 17, 255 - d06[1])
            d19 = (label * 24 + 18, 255 - d07[1])
            d20 = (label * 24 + 19, 255 - d08[1])
            d21 = (label * 24 + 20, 255 - d09[1])
            d22 = (label * 24 + 21, 255 - d10[1])
            d23 = (label * 24 + 22, 255 - d11[1])
            d24 = (label * 24 + 23, 255 - d12[1])

            dataset.extend([
                d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12,
                d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24,
            ])

        return dataset

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.dataset))
        label, base_img = self.dataset[idx]
        img = D.imresize(
            base_img.copy(),
            size=self.image_size,
            interpolation=self.interpolation
        )

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
        'Passport': 4,
        'ResidentIDCardFront': 5,
        'ResidentIDCardBack': 6,
        'VehicleLicense': 7
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
        fs = D.get_files(self.root, suffix=['.jpg', '.png', '.jpeg'])
        return [(f.parent.name, f) for f in fs]

    def __getitem__(self, idx):
        label, file = self.dataset[idx]
        img = D.imread(file)
        img = D.imresize(img, self.image_size)
        if self.return_tensor:
            img = img.transpose(2, 0, 1).astype('float32') / 255.0
        return img, self.label_mapper[label]
