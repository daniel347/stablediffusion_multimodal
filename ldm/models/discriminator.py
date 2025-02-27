# Borrowed from DCGAN
# https://github.com/Lornatang/DCGAN-PyTorch/blob/master/model.py
from torch import nn, Tensor
import torch
from ..util import instantiate_from_config, remove_module_prefix


class Discriminator(nn.Module):
    def __init__(self, in_size) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is 1 x 64 x 64
            nn.Conv2d(1, 64, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 32 x 32
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 16 x 16
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 8 x 8
            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 4 x 4
            # This final layer averages all activations
            nn.Conv2d(512, 1, (in_size//16, in_size//16), (1, 1), (0, 0), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        out = torch.flatten(out, 1)

        return out


class MMDiscriminator(nn.Module):
    def __init__(self, disc_params) -> None:
        super(MMDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([instantiate_from_config(params) for params in disc_params])

    def forward(self, X):
        Y = []
        for m, x in enumerate(X):
            Y.append(self.discriminators[m](x))

        return Y