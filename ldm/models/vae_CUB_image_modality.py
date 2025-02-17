# CUB Image-Captions Unimodal VAE Image model specification
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import v2

from utils import Constants
from .base_vae import VAE
from .encoder_decoder_blocks.resnet_cub_image import EncoderImg, DecoderImg
from .dataset_NYU_Depth import NYUDepth


class CUB_Image(VAE):
    """ Unimodal VAE subclass for Image modality CUB Image-Captions experiment """

    def __init__(self, params, vae_params=None):
        super(CUB_Image, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,                 # prior
            dist.Laplace,                                                                       # likelihood
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,                 # posterior
            EncoderImg(params.latent_dim_w, params.latent_dim_z, dist=params.priorposterior, in_channels=vae_params.dataSize[0]),   # Encoder model
            DecoderImg(params.latent_dim_u, out_channels=vae_params.dataSize[0]),               # Decoder model
            params                                                                              # Params (args passed to main)
        )
        self._pw_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=True)  # It is important that this log-variance vector is learnable (see paper)
        ])

        self.modelName = 'cubI' if vae_params is None else vae_params.modelName
        self.dataSize = torch.Size([3, 64, 64]) if vae_params is None else vae_params.dataSize
        self.modality = "image" if vae_params is None else vae_params.modality

        self.llik_scaling = 1.
        self.params = params

    @property
    def pw_params_aux(self):
        """

        Returns: Parameters of prior auxiliary distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params_aux[0], F.softplus(self._pw_params_aux[1]) + Constants.eta
        else:
            return self._pw_params_aux[0], F.softmax(self._pw_params_aux[1], dim=-1) * self._pw_params_aux[1].size(
                -1) + Constants.eta

    # remember that when combining with captions, this should be x10
    def getDataLoaders(self, batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        tx = v2.Compose([v2.Resize([64, 64]), v2.ToTensor(), v2.ToDtype(torch.float32, scale=True)])
        train_loader = torch.utils.data.DataLoader(
            NYUDepth(self.params.datadir, transform=tx, train=True, modality=self.modality),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            NYUDepth(self.params.datadir, transform=tx, train=False, modality=self.modality),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader


