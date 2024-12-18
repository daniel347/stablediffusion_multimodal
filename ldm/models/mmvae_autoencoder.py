# Base MMVAEplus class definition
import torch
import torch.nn as nn
from ..util import instantiate_from_config
from torch.distributions import Normal, Laplace
import torch.nn.functional as F


class MMVAEplus(nn.Module):
    """
    MMVAEplus class definition.
    """
    def __init__(self, prior_dist, latent_channel_z, latent_channel_w, vae_params=None):
        super(MMVAEplus, self).__init__()
        if isinstance(prior_dist, str):
            if prior_dist == 'normal':
                prior_dist = Normal
            elif prior_dist == 'laplace':
                prior_dist = Laplace
            else:
                raise ValueError('Invalid prior distribution')

        self.pz = prior_dist # Prior distribution
        self.pw = prior_dist
        self.vaes = nn.ModuleList([instantiate_from_config(params) for params in vae_params])  # List of unimodal VAEs (one for each modality)
        self.modelName = None  # Filled-in in subclass

        self.w_channels = latent_channel_w
        self.z_channels = latent_channel_z

        w_shape = (vae_params[0]["params"]["ddconfig"]["w_channels"], vae_params[0]["params"]["embed_spatial_dim"], vae_params[0]["params"]["embed_spatial_dim"])
        z_shape = (vae_params[0]["params"]["ddconfig"]["z_channels"] - vae_params[0]["params"]["ddconfig"]["w_channels"],
                   vae_params[0]["params"]["embed_spatial_dim"],
                   vae_params[0]["params"]["embed_spatial_dim"])

        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, *w_shape), requires_grad=False),
            nn.Parameter(torch.zeros(1, *w_shape), requires_grad=False)  # It is important that this log-variance vector is learnable (see paper)
        ])
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, *z_shape), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, *z_shape), requires_grad=False)  # logvar
        ])

        self.latent_channel_z = latent_channel_z
        self.latent_channel_w = latent_channel_w

    @staticmethod
    def getDataSets(batch_size, shuffle=True, device="cuda"):
        # Handle getting individual datasets appropriately in sub-class
        raise NotImplementedError

    @property
    def pw_params(self):
        """ Returns: Parameters of prior distribution for modality-specific latent code """
        if self.pw == Normal:
            return self._pw_params[0], F.softplus(self._pw_params[1]) + 1e-20
        else:
            return self._pw_params[0], F.softmax(self._pw_params[1], dim=(1, 2, 3)) * self._pw_params[1].size(-1) + 1e-20

    @property
    def pz_params(self):
        """ Returns: Parameters of prior distribution for shared latent code """
        if self.pw == Normal:
            return self._pz_params[0], F.softplus(self._pz_params[1]) + 1e-20
        else:
            return self._pz_params[0], F.softmax(self._pz_params[1], dim=(1, 2, 3)) * self._pz_params[1].size(-1) + 1e-20

    def forward(self, x, K=1):
        # print(f"model forward received x: {[xi.shape for xi in x]} on device {[xi.device for xi in x]}")
        """
        Forward function.
        Input:
            - x: list of data samples for each modality
            - K: number of samples for reparameterization in latent space

        Returns:
            - qu_xs: List of encoding distributions (one per encoder)
            - px_us: Matrix of self- and cross- reconstructions. px_zs[m][n] contains
                    m --> n  reconstruction.
            - uss: List of latent codes, one for each modality. uss[m] contains latents inferred
                   from modality m. Note there latents are the concatenation of private and shared latents.
        """
        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K) # Get Encoding dist, Decoding dist, Latents for unimodal VAE m modality
            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix
        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    _, z_e = torch.split(us, [self.latent_channel_w, self.latent_channel_z], dim=2)
                    # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality m
                    pw = vae.pw(*vae.pw_params_aux)
                    latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2).to(x[0].device)

                    # Combine shared and resampled private latents
                    us_combined = torch.cat((latents_w, z_e), dim=2)
                    # Get cross-reconstruction likelihood
                    px_us[e][d] = vae.decode(us_combined)
        return qu_xs, px_us, uss


    def generate_unconditional(self, N):
        """
        Unconditional generation.
        Args:
            N: Number of samples to generate.
        Returns:
            Generations
        """
        with torch.no_grad():
            data = []
            # Sample N shared latents
            pz = self.pz(*self.pz_params)
            latents_z = pz.rsample(torch.Size([N]))
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z.size()[0]])
                latents = torch.cat((latents_w, latents_z), dim=2)
                px_u = vae.decode(latents)
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality


    # def reconstruct(self, data):
    #     """
    #     Test-time reconstruction
    #     Args:
    #         data: Input
    #
    #     Returns:
    #         Reconstructions
    #     """
    #     with torch.no_grad():
    #         _, px_zs, _ = self.forward(data)
    #         # cross-modal matrix of reconstructions
    #         recons = [[get_mean(px_z) for px_z in r] for r in px_zs]
    #     return recons

    def self_and_cross_modal_generation_forward(self, data, K=1):
        """
        Test-time self- and cross-model generation forward function.
        Args:
            data: Input

        Returns:
            Unimodal encoding distribution, Matrix of self- and cross-modal reconstruction distrubutions, Latent embeddings

        """
        qu_xs, uss = [], []
        # initialise cross-modal matrix
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(data[m], K=K)
            qu_xs.append(qu_x)
            uss.append(us)
            px_us[m][m] = px_u  # fill-in diagonal
        for e, us in enumerate(uss):
            latents_w, latents_z = torch.split(us, [self.latent_channel_w, self.latent_channel_z], dim=2)
            for d, vae in enumerate(self.vaes):
                mean_w, scale_w = self.pw_params
                # Tune modality-specific std prior
                # scale_w = factor * scale_w
                pw = self.pw(mean_w, scale_w)
                latents_w_new = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                us_new = torch.cat((latents_w_new, latents_z), dim=2)
                if e != d:  # fill-in off-diagonal
                    px_us[e][d] = vae.decode(us_new)
        return qu_xs, px_us, uss

    def self_and_cross_modal_generation(self, data):
        """
        Test-time self- and cross-reconstruction.
        Args:
            data: Input

        Returns:
            Matrix of self- and cross-modal reconstructions

        """
        with torch.no_grad():
            _, px_us, _ = self.self_and_cross_modal_generation_forward(data)
            # ------------------------------------------------
            # cross-modal matrix of reconstructions
            recons = [[px_u.mean for px_u in r] for r in px_us]
        return recons
