from typing import Any, Callable, Tuple, Union

import albumentations as A
import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import numpy as np
from docsaidkit import INTER, Path

DIR = D.get_curdir(__file__)


class DefaultImageAug:

    def __init__(self, p=0.5):

        self.aug = A.Compose([

            DT.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=15,
            ),

            A.OneOf([
                A.Spatter(mode='mud'),
                A.GaussNoise(),
                A.ISONoise(),
                A.MotionBlur(),
                A.Defocus(),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=32,
                    max_width=32,
                    min_height=5,
                    min_width=5,
                    fill_value=255,
                ),
            ], p=p),

            A.OneOf([
                A.Equalize(),
                A.ColorJitter(),
            ]),

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
        fs = D.get_files(self.root, suffix=['.jpg', '.png', '.jpeg'])

        dataset = []
        for label, f in D.Tqdm(enumerate(fs)):
            img = D.imread(f)

            d1 = (label * 12, img)
            d2 = (label * 12 + 1, D.imrotate(img, 90))
            d3 = (label * 12 + 2, D.imrotate(img, 180))
            d4 = (label * 12 + 3, D.imrotate(img, 270))
            d5 = (label * 12 + 4, cv2.flip(img, 0))
            d6 = (label * 12 + 5, cv2.flip(D.imrotate(img, 90), 0))
            d7 = (label * 12 + 6, cv2.flip(D.imrotate(img, 180), 0))
            d8 = (label * 12 + 7, cv2.flip(D.imrotate(img, 270), 0))
            d9 = (label * 12 + 8, cv2.flip(img, 1))
            d10 = (label * 12 + 9, cv2.flip(D.imrotate(img, 90), 1))
            d11 = (label * 12 + 10, cv2.flip(D.imrotate(img, 180), 1))
            d12 = (label * 12 + 11, cv2.flip(D.imrotate(img, 270), 1))

            dataset.extend([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12])

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
