# Base MMVAEplus class definition
import torch
import torch.nn as nn
from ..util import instantiate_from_config, remove_module_prefix
from torch.distributions import Normal, Laplace
import torch.nn.functional as F
from ..data.dataset_CelebAMask import MMCelebA

class IndSepAE(nn.Module):
    """
    MMVAEplus class definition.
    """
    def __init__(self, prior_dist, latent_channel_u, vae_params=None, ckpt_path=None):
        super(IndSepAE, self).__init__()
        if isinstance(prior_dist, str):
            if prior_dist == 'normal':
                prior_dist = Normal
            elif prior_dist == 'laplace':
                prior_dist = Laplace
            else:
                raise ValueError('Invalid prior distribution')

        self.pu = prior_dist # Prior distribution
        self.vaes = nn.ModuleList([instantiate_from_config(params) for params in vae_params])  # List of unimodal VAEs (one for each modality)
        self.modelName = None  # Filled-in in subclass

        self.u_channels = latent_channel_u

        u_shape = (vae_params[0]["params"]["ddconfig"]["z_channels"], vae_params[0]["params"]["embed_spatial_dim"], vae_params[0]["params"]["embed_spatial_dim"])

        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, *u_shape), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, *u_shape), requires_grad=False)  # logvar
        ])

        if ckpt_path is not None:
            print("Loading MMVAE checkpoint from {params.ckpt_path}")
            ckpt = torch.load(ckpt_path)
            ckpt = remove_module_prefix(ckpt)
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            if len(missing) > 0:
                print(f"Missing Keys:\n {missing}")
            if len(unexpected) > 0:
                print(f"\nUnexpected Keys:\n {unexpected}")

        self.can_cross_generate = False
        self.can_sample = False

    @staticmethod
    def getDataSets(batch_size, shuffle=True, device="cuda"):
        # Handle getting individual datasets appropriately in sub-class
        raise NotImplementedError

    @property
    def pu_params(self):
        """ Returns: Parameters of prior distribution for modality-specific latent code """
        if self.pu == Normal:
            return self._pu_params[0], F.softplus(self._pu_params[1]) + 1e-20
        else:
            return self._pu_params[0], F.softmax(self._pu_params[1], dim=(1, 2, 3)) * self._pu_params[1].size(-1) + 1e-20


    def forward(self, x, K=1):
        qu_xs, uss, px_us = [], [], []

        for m, vae in enumerate(self.vaes):
            qu_x = vae.encode(x[m])
            us = vae.sample(qu_x, K=K)
            px_u = vae.decode(us)

            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            px_us.append(px_u)  # Fill-in self-reconstructions in the matrix

        return qu_xs, px_us, uss

    def encode(self, data):
        # Encode data into z and W
        U = []

        for dm, vae in zip(data, self.vaes):
            qu_x = vae.encode(dm)
            U.append(qu_x.mean)


        full_latent = torch.concat(U, dim=1)
        return full_latent

    def decode(self, full_latent):
        # convert z and W to image and depth
        U = torch.split(full_latent, [self.u_channels for _ in range(len(self.vaes))], dim=1)

        recons = []
        for vae, u in zip(self.vaes, U):
            recons.append(vae.decode(u))

        return recons