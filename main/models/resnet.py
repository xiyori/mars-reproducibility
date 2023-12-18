from torch import nn
from typing import Sequence

from mars import MARS
from tensorized_models import TTConv2d
from .mars_config import MarsConfig


class Block(nn.Module):
    def __init__(self, config: MarsConfig, num_channels: int, stride: int,
                 pi: float, alpha: float):
        super().__init__()

        self.num_channels = num_channels
        self.stride = stride

        self.norm1 = nn.Sequential(
            nn.BatchNorm2d(self.num_channels // self.stride),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            MARS(
                TTConv2d(
                    self.num_channels // self.stride,
                    self.num_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    bias=False,
                    d=config.d,
                    tt_rank=config.tt_rank,
                    auto_shapes=config.auto_shapes,
                    shape=config.shape
                ),
                pi=pi,
                alpha=alpha
            ) if config.enabled and self.stride == 1 else nn.Conv2d(
                self.num_channels // self.stride,
                self.num_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(),
            MARS(
                TTConv2d(
                    self.num_channels,
                    self.num_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    d=config.d,
                    tt_rank=config.tt_rank,
                    auto_shapes=config.auto_shapes,
                    shape=config.shape
                ),
                pi=pi,
                alpha=alpha
            ) if config.enabled else nn.Conv2d(
                self.num_channels,
                self.num_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        if self.stride != 1:
            assert self.stride == 2

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    self.num_channels // self.stride,
                    self.num_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False
                ),
                nn.BatchNorm2d(self.num_channels)
            )

    def forward(self, x):
        normalized = self.norm1(x)
        residual = self.shortcut(normalized) if hasattr(self, 'shortcut') else x
        x = self.layers(normalized) + residual
        return x


def make_block_group(config: MarsConfig, num_channels: int, num_blocks: int, stride: int,
                     pi: float, alpha: float):
    assert num_blocks > 0
    strides = [stride] + [1] * (num_blocks - 1)
    return nn.Sequential(
        *[Block(config, num_channels, stride, pi, alpha) for stride in strides]
    )


class ResNet(nn.Module):
    def __init__(self, config, pi: float, alpha: float):
        super().__init__()
        self.config = config

        self.blocks_per_group: Sequence[int] = config.blocks_per_group
        self.num_classes: int = config.num_classes
        self.width: int = config.width
        channels = self.width

        self.conv1 = nn.Conv2d(
            3, channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        channels_per_group = (channels, 2 * channels, 4 * channels, 8 * channels)[:len(self.blocks_per_group)]
        strides_per_group = (1, 2, 2, 2)[:len(self.blocks_per_group)]
        self.blockgroups = nn.Sequential(*[
            make_block_group(mconfig, c, b, s, pi, alpha)
            for mconfig, c, b, s in zip(
                config.mars_configs, channels_per_group, self.blocks_per_group, strides_per_group
            )
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(channels_per_group[-1], self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blockgroups(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
