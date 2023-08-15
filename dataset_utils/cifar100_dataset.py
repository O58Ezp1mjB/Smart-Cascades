import os
import logging
import torch
from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets import CIFAR100
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pickle
from PIL import Image
import glob
import wandb
from table_reader import table_read

DEFAULT_TIMM_OUTPUT_PATH = "./output/train"

_logger = logging.getLogger('dataset')

class LTC_CIFAR100(CIFAR100):

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False,
                 expensive_prediction_path: str = None):
        super(LTC_CIFAR100, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        self.expensive_prediction = None
        if (expensive_prediction_path is not None) and (not expensive_prediction_path == ""):
            _logger.info("Reading .bz2 or .csv file of expensive_prediction.")

            if expensive_prediction_path.startswith("wandb://"):
                api = wandb.Api()
                run = api.run(expensive_prediction_path[8:])
                # check whether run already existed
                if os.path.isdir(os.path.join(DEFAULT_TIMM_OUTPUT_PATH, run.name)):
                    _logger.warning(f"Folder {os.path.join(DEFAULT_TIMM_OUTPUT_PATH, run.name)} already existed.")
                    expensive_prediction_path = os.path.join(DEFAULT_TIMM_OUTPUT_PATH, run.name)
                else:
                    # download from wandb
                    for file in run.files():
                        if file.name.endswith("bz2"):
                            file.download()
                    expensive_prediction_path = os.path.join(DEFAULT_TIMM_OUTPUT_PATH, run.name)
            _logger.info(f"Try to search .bz2 or .csv file under {expensive_prediction_path}")
            # use bz2 if existed
            if len(glob.glob(os.path.join(expensive_prediction_path, "*.bz2"))) > 0:
                bz2_file_list = glob.glob(os.path.join(expensive_prediction_path, "*.bz2"))
                self.expensive_prediction = table_read(bz2_file_list)
            # use csv if bz2 not existed
            elif len(glob.glob(os.path.join(expensive_prediction_path, "*.csv"))) > 0:
                csv_file_list = glob.glob(os.path.join(expensive_prediction_path, "*.csv"))
                self.expensive_prediction = table_read(csv_file_list)
            # neither csv, bz2 found
            else:
                raise ValueError("Get error while reading .bz2 or .csv file in expensive_prediction_path")

    # copy from torchvision/datasets/cifar.py
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        exp_prediction = [-1]
        if self.expensive_prediction is not None:
            pred = self.expensive_prediction.get(index, None)
            exp_prediction = pred["predict"] if pred else [-1]

        return img, target, exp_prediction
