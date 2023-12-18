from torch import nn
import torch.nn.functional as F

from mars import MARS
from tensorized_models import TuckerConv2d, FactorizedLinear


class LeNet(nn.Module):
    def __init__(self, config, pi: float, alpha: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = MARS(
            TuckerConv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                rank=20
            ),
            pi=pi,
            alpha=alpha
        ) if config.mars_enabled else nn.Conv2d(6, 16, 5)
        self.fc1 = MARS(
            FactorizedLinear(
                in_channels=16 * 5 * 5,
                out_channels=120,
                rank=100
            ),
            pi=pi,
            alpha=alpha
        ) if config.mars_enabled else nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
