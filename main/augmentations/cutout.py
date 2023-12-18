import numpy as np
import torch
from typing import Sequence
from torch.utils.data import Dataset


class CutoutWrapper(Dataset):
    """Cutout augmentation wrapper.

    Args:
        dataset (Dataset): torch dataset loader
        size (int): hole size
        p (float): probability of applying augmentation

    """

    def __init__(
            self,
            dataset,
            size,
            p=0.5,
    ):
        self.dataset = dataset
        self.size = size
        self.p = p

    def __getitem__(self, i):
        img, lbl = self.dataset[i]
        extra = None
        if isinstance(img, Sequence):
            img, extra = img[0], img[1:]

        if np.random.rand() < self.p:
            mask = torch.ones_like(img)

            y_center = np.random.randint(img.shape[1])
            x_center = np.random.randint(img.shape[2])

            height = self.size  # np.random.randint(self.size + 1)
            width = self.size  # np.random.randint(self.size + 1)

            top = np.clip(y_center - height // 2, 0, img.shape[1])
            bottom = np.clip(y_center + height // 2, 0, img.shape[1])
            left = np.clip(x_center - width // 2, 0, img.shape[2])
            right = np.clip(x_center + width // 2, 0, img.shape[2])

            mask[:, top:bottom, left:right] = 0
            img = img * mask

        if extra is None:
            return img, lbl
        return (img, *extra), lbl

    def __len__(self):
        return len(self.dataset)
