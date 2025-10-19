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


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm=False, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.Tanh(),
            ]
            if norm:
                sequence += [norm_layer(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.Tanh(),
        ]
        if norm:
            sequence += [norm_layer(ndf * nf_mult)]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                               stride=1, padding=padw), nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class MMDiscriminator(nn.Module):
    def __init__(self, disc_params) -> None:
        super(MMDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([instantiate_from_config(params) for params in disc_params])

    def forward(self, X):
        print(X)
        Y = []
        for m, x in enumerate(X):
            if isinstance(x, torch.distributions.Distribution):
                x = x.mode
            Y.append(self.discriminators[m](x))

        return Y

class MMJointDiscriminator(nn.Module):
    def __init__(self, disc_params) -> None:
        super(MMJointDiscriminator, self).__init__()
        self.discriminator = instantiate_from_config(disc_params)

    def forward(self, X):
        if isinstance(X[0], torch.distributions.Distribution):
            X = [x.mode for x in X]
        x_concat = torch.cat(X, dim=-3)
        if len(x_concat.shape) == 5:
            x_concat = x_concat.squeeze(0)
            return [self.discriminator(x_concat).unsqueeze(0)]
        else:
            return [self.discriminator(x_concat)]