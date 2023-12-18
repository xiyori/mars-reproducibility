import torch
import numpy as np

from typing import Sequence
from torch.utils.data import Dataset


class MixupWrapper(Dataset):
    """Muxup augmentation wrapper.

    Args:
        dataset (Dataset): torch dataset loader
        p (float): probability of applying augmentation
        alpha (float): beta distribution param

    """

    def __init__(
            self,
            dataset,
            alpha,
            p=0.5
    ):
        self.dataset = dataset
        self.alpha = alpha
        self.p = p

    def __getitem__(self, i):
        img, lbl = self.dataset[i]
        extra = None
        if isinstance(img, Sequence):
            img, extra = img[0], img[1:]

        if np.random.rand() < self.p:
            img_mix, lbl_mix = self.dataset[np.random.randint(len(self))]
            if extra is not None:
                img_mix, extra_mix = img_mix[0], img_mix[1:]
            lam = torch.tensor(
                np.random.beta(self.alpha, self.alpha),
                dtype=torch.float32
            )
            img = img * lam + img_mix * (1 - lam)
            lbl = (lam, lbl, lbl_mix)
            if extra is not None:
                for e, e_mix in zip(extra, extra_mix):
                    e[0] = 1 - lam
                    e[1:3] = e_mix[3:]
        else:
            lbl = lbl

        if extra is None:
            return img, lbl
        return (img, *extra), lbl

    def __len__(self):
        return len(self.dataset)
