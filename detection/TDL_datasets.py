from argparse import ArgumentParser
from typing import Optional, Union, Any, Callable, Tuple, List

import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import SVHN
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    SVHN = None


class SVHNDataModule1(VisionDataModule):
    """
    Specs:
        - 10 classes (1 per class)
        - Each image is (3 x 32 x 32)

    Standard SVHN, train, val, test splits and transforms

    Transforms::

        transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])

    Example::

        from pl_bolts.datamodules import CIFAR10DataModule

        dm = CIFAR10DataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "svhn"
    dataset_cls = SVHN
    dims = (3, 32, 32)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=73_257)
        return train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def default_transforms(self) -> Callable:
        svhn_transforms = transform_lib.Compose([transform_lib.ToTensor()])
        return svhn_transforms

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=32)

        return parser


class SVHNDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_transform=None, test_transform=None, val_transform=None,
                 batch_size=32, num_workers=0, val_split=0.15, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

    def prepare_data(self):
        SVHN(self.data_dir, split='train', download=True)
        SVHN(self.data_dir, split='test', download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dataset_train = SVHN(self.data_dir, split="train", transform=self.train_transform)
            dataset_val = SVHN(self.data_dir, split="train", transform=self.val_transform)

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

            # self.train_dataset = SVHN(self.data_dir, split='train',
            #                           transform=self.train_transform)
            # self.val_dataset = SVHN(self.data_dir, split='test',
            #                         transform=self.test_transform)
        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transform is None else self.test_transform
            self.dataset_test = SVHN(
                self.data_dir, split='test', transform=test_transforms
            )

    def default_transforms(self) -> Callable:
        cf10_transforms = transform_lib.Compose([transform_lib.ToTensor()])
        return cf10_transforms

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(self.seed))

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits
