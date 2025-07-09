# Base MMVAEplus class definition
import torch
import torch.nn as nn

from ..modules.distributions.distributions import DiagonalGaussianDistribution
from ..util import instantiate_from_config, remove_module_prefix
from torch.distributions import Normal, Laplace
import torch.nn.functional as F
from ..data.dataset_CelebAMask import MMCelebA
import torch.distributions as dist
from omegaconf import OmegaConf

class NonDisentanglingAE(nn.Module):
    """
    MMVAEplus class definition.
    """
    def __init__(self, prior_dist, latent_channel_u, lik_dist_splits=None, lik_dists=None, final_acts=None, llik_scales=None, seperable=True, vae_params=None, ckpt_path=None, scale_per_channel=False):
        super(NonDisentanglingAE, self).__init__()
        if isinstance(prior_dist, str):
            if prior_dist == 'normal':
                prior_dist = Normal
            elif prior_dist == 'laplace':
                prior_dist = Laplace
            else:
                raise ValueError('Invalid prior distribution')

        if lik_dists is not None:
            lik_dists = OmegaConf.to_container(lik_dists)
            for i in range(len(lik_dists)):
                if isinstance(lik_dists[i], str):
                    lik_dists[i] = getattr(dist, lik_dists[i])

        if final_acts is not None:
            final_acts = OmegaConf.to_container(final_acts)
            for i in range(len(final_acts)):
                if isinstance(final_acts[i], str):
                    if final_acts[i] == "identity":
                        final_acts[i] = lambda x: x
                    else:
                        final_acts[i] = getattr(torch.nn.functional, final_acts[i])

        self.llik_scales = llik_scales
        if self.llik_scales is None:
            self.llik_scales = [1.0 for _ in lik_dists]

        self.pu = prior_dist # Prior distribution
        assert (len(vae_params) == 1) or seperable, "When seperable=False this class expects a single multimodal VAE only"
        assert (lik_dist_splits is not None and lik_dists is not None) or seperable, "When seperable=False this class expects likelihood distribution outputs"
        if not seperable:
            vae_params[0]["params"]["return_lik_dist"] = False  # we handle potentially multiple distributions in this class
        self.vaes = nn.ModuleList([instantiate_from_config(params) for params in vae_params])  # List of unimodal VAEs (one for each modality)
        self.modelName = None  # Filled-in in subclass

        self.u_channels = latent_channel_u
        if scale_per_channel:
            self.latent_mean = nn.Parameter(torch.zeros(self.u_channels))
            self.latent_scale = nn.Parameter(torch.ones(self.u_channels) * 1.0 / 0.18215)
        else:
            self.latent_mean = nn.Parameter(torch.tensor(0.0))
            self.latent_scale = nn.Parameter(torch.tensor(1.0 / 0.18215))

        u_shape = (vae_params[0]["params"]["ddconfig"]["z_channels"], vae_params[0]["params"]["embed_spatial_dim"], vae_params[0]["params"]["embed_spatial_dim"])

        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, *u_shape), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, *u_shape), requires_grad=False)  # logvar
        ])

        if ckpt_path is not None:
            print("Loading MMVAE checkpoint from {params.ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            ckpt = remove_module_prefix(ckpt)
            if "latent_mean" in ckpt and ckpt["latent_mean"].shape != self.latent_mean.shape:
                del ckpt["latent_mean"]
            if "latent_scale" in ckpt and ckpt["latent_scale"].shape != self.latent_scale.shape:
                del ckpt["latent_scale"]
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            if len(missing) > 0:
                print(f"Missing Keys:\n {missing}")
            if len(unexpected) > 0:
                print(f"\nUnexpected Keys:\n {unexpected}")

        self.can_cross_generate = False
        self.can_sample = False
        self.factorised_latent_space = False

        self.seperable = seperable

        if not self.seperable:
            self.lik_dist_splits = OmegaConf.to_container(lik_dist_splits)
            self.lik_dists = lik_dists
            self.final_acts = final_acts


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

        if not self.seperable:
            X = torch.cat(x, dim=1)
            qu_xs = [self.vaes[0].encode(X)]
            uss = [self.vaes[0].sample(qu_xs[0], K=K)]
            px_us = self.vaes[0].decode(uss[0])
            px_us = self.create_likelihood_dists(px_us)
        else:
            for m, vae in enumerate(self.vaes):
                qu_x = vae.encode(x[m])
                us = vae.sample(qu_x, K=K)
                px_u = vae.decode(us)

                qu_xs.append(qu_x) # Append encoding distribution to list
                uss.append(us) # Append latents to list
                px_us.append(px_u)  # Fill-in self-reconstructions in the matrix

        return qu_xs, px_us, uss

    def encode(self, data, return_dist=True, **kwargs):
        # Encode data into z and W
        if not self.seperable:
            X = torch.cat(data, dim=1)
            qu_x = self.vaes[0].encode(X)
            if return_dist:
                return qu_x
            else:
                return qu_x.mean
        else:
            U = []
            for dm, vae in zip(data, self.vaes):
                qu_x = vae.encode(dm)
                U.append(qu_x)

            if return_dist:
                return U
            else:
                return torch.concat([u.mean for u in U], dim=1)

    def decode(self, full_latent, **kwargs):
        # convert z and W to image and depth
        U = torch.split(full_latent, [self.u_channels for _ in range(len(self.vaes))], dim=1)

        recons = []
        for vae, u in zip(self.vaes, U):
            recons.append(vae.decode(u))

        if not self.seperable:
            recons = self.create_likelihood_dists(recons[0])

        return recons

    def create_likelihood_dists(self, px_us):
        ch_dim = 2 if len(px_us.shape) == 5 else 1
        recons_split = torch.split(px_us, self.lik_dist_splits, dim=ch_dim)
        scale = torch.tensor(0.01).to(px_us.device)

        px_us_dists = []

        for ld, act, split in zip(self.lik_dists, self.final_acts, recons_split):
            if ld == dist.Categorical or ld == dist.Bernoulli:
                px_us_dists.append(ld(act(split)))
            else:
                px_us_dists.append(ld(act(split), scale=scale))

        return px_us_dists

    def scale_latents(self, latents, reverse=False):
        if reverse:
            if len(self.latent_scale.shape) == 1:
                return (latents * self.latent_scale[None, :, None, None]) + self.latent_mean[None, :, None, None]
            else:
                return (latents * self.latent_scale) + self.latent_mean
        else:
            if len(self.latent_scale.shape) == 1:
                return (latents - self.latent_mean[None, :, None, None]) / self.latent_scale[None, :, None, None]
            else:
                return (latents - self.latent_mean) / self.latent_scale
