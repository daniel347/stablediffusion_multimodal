# Base MMVAEplus class definition
import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.model import ResnetBlock
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

class ZCombinerNet(nn.Module):

    def __init__(self, z_dim, nz, ch, dist_input_output, nblocks):
        super(ZCombinerNet, self).__init__()

        self.dist_input_output = dist_input_output
        mult = 2 if dist_input_output else 1

        self.in_conv = nn.Conv2d(z_dim * nz * mult, ch, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([ResnetBlock(in_channels=ch,
                                                 out_channels=ch,
                                                 dropout=0.1,
                                                 temb_channels=0) for _ in range(nblocks)])

        self.out_conv = nn.Conv2d(ch, z_dim * mult, kernel_size=3, stride=1, padding=1)

    def forward(self, zs):
        if isinstance(zs[0], DiagonalGaussianDistribution):
            mu_zs = [z.mean for z in zs]
            logvar_zs = [z.logvar for z in zs]
            z = torch.cat(mu_zs + logvar_zs, dim=1)
        else:
            z = torch.cat(zs, dim=1)

        z = self.in_conv(z)
        for block in self.blocks:
            z = block(z, None)

        z = self.out_conv(z)
        if self.dist_input_output:
            posterior = DiagonalGaussianDistribution(z)
            return posterior
        return z


