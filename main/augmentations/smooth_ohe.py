import numpy as np
from typing import Sequence

import torch
from torch.utils.data import Dataset


class SmoothOHEWrapper(Dataset):
    """Smooth OHE wrapper.

    Args:
        dataset (Dataset): torch dataset loader
        *augs: augmentation wrappers

    """

    def __init__(
            self,
            dataset,
            n_classes
    ):
        self.dataset = dataset
        self.n_classes = n_classes

    def __getitem__(self, i):
        img, lbl = self.dataset[i]

        ohe_lbl = torch.zeros(self.n_classes)

        if isinstance(lbl, Sequence):
            lam, lbl, lbl_mix = lbl
            ohe_lbl[lbl] = lam
            ohe_lbl[lbl_mix] = 1 - lam
        else:
            ohe_lbl[lbl] = 1

        return img, ohe_lbl

    def __len__(self):
        return len(self.dataset)
