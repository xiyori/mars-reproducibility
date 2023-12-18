import numpy as np
from torch.utils.data import Dataset


class CombineWrapper(Dataset):
    """Combine augmentation wrapper.

    Args:
        dataset (Dataset): torch dataset loader
        *augs: augmentation wrappers

    """

    def __init__(
            self,
            *augs
    ):
        self.augs = augs
        self.probs = np.array([aug.p for aug in self.augs])
        self.cum_probs = np.cumsum(self.probs / self.probs.sum())
        for aug in self.augs:
            aug.p = 1.

    def __getitem__(self, i):
        unif = np.random.rand()
        for aug, prob in zip(self.augs, self.cum_probs):
            if unif < prob:
                return aug[i]

        return self.augs[-1][i]

    def __len__(self):
        return len(self.augs[0])
