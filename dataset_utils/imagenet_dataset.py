import os
import pickle

from torchvision.datasets.imagenet import ImageNet
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image


class LTC_ImageNet(ImageNet):

    def __init__(self, root: str, split: str = "train", expensive_prediction_path: str = None, **kwargs: Any):
        super(LTC_ImageNet, self).__init__(root=root, split=split, **kwargs)
        self.expensive_prediction = None
        if expensive_prediction_path is not None and \
                not expensive_prediction_path == "" and \
                os.path.isfile(os.path.join(root, "expensive_prediction", expensive_prediction_path)):
            with open(os.path.join(root, "expensive_prediction", expensive_prediction_path), "rb") as f:
                self.expensive_prediction = pickle.load(f, encoding="latin1")

    # copy from torchvision/datasets/folder.py L220
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        exp_prediction = self.expensive_prediction[index] if self.expensive_prediction else [-1]

        return sample, target, exp_prediction

