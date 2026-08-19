import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from omegaconf import OmegaConf

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma

import torch.nn as nn

import torch.distributions as dist


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 embed_spatial_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False,
                 prior_dist=dist.Normal,
                 likelihood_dist=dist.Laplace,
                 post_dist=dist.Normal,
                 conform_to_3_channel=False,
                 is_semseg=False,
                 llik_scaling=1.0,
                 return_lik_dist=True,
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key

        if isinstance(likelihood_dist, str): likelihood_dist = getattr(dist, likelihood_dist)
        if isinstance(prior_dist, str): prior_dist = getattr(dist, prior_dist)
        if isinstance(post_dist, str): post_dist = getattr(dist, post_dist)

        if not isinstance(ddconfig, dict):
            ddconfig = OmegaConf.to_container(ddconfig)

        self.is_semseg = is_semseg

        self.conform_to_3_channel = conform_to_3_channel
        if is_semseg:
            ddconfig["is_semseg"] = True

            assert likelihood_dist == dist.Categorical or likelihood_dist == dist.Bernoulli, \
                "Semantic segmentation requires a Categorical (one class per pixel) or Bernoilli (multiple classes per pixel) likelihood distribution"

        if conform_to_3_channel:
            assert (not is_semseg), "You can only conform to 3 channels if your input is less than 3 channels (i.e. no semantic segmentation)"
            self.input_expansion_factor = int(3 / ddconfig["in_channels"])
            ddconfig["in_channels"] = 3
            ddconfig["out_ch"] = 3

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.qu_x = post_dist
        self.px_u = likelihood_dist
        self.pw = prior_dist

        self.llik_scaling = llik_scaling

        self._qu_x_params = None
        self._pw_params_aux = None

        self._pw_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ddconfig["w_channels"], embed_spatial_dim, embed_spatial_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, ddconfig["w_channels"], embed_spatial_dim, embed_spatial_dim), requires_grad=True)  # It is important that this log-variance vector is learnable (see paper)
        ])

        self.return_lik_dist = return_lik_dist

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            print(f"Loading AutoencoderKL checkpoint from {ckpt_path}")
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def expand_input_channels(self, x):
        return x.repeat(1, self.input_expansion_factor, 1, 1)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        M, U = self.load_state_dict(sd, strict=False)
        for m in M:
            print(f"missing key: {m}")
        for u in U:
            print(f"unused key: {u}")
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    @property
    def pw_params_aux(self):
        """ Returns: Parameters of prior distribution for the mode specific latent code """
        if self.pw == dist.Normal:
            return self._pw_params_aux[0], F.softplus(self._pw_params_aux[1]) + 1e-20
        else:
            return self._pw_params_aux[0], F.softmax(self._pw_params_aux[1], dim=(1, 2, 3)) * self._pw_params_aux[1].size(-1) + 1e-20

    @property
    def qu_x_params(self):
        """Get encoding distribution parameters (already adapted for the specific distribution at the end of the Encoder class)"""
        if self._qu_x_params is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qu_x_params

    def encode(self, x):
        if self.conform_to_3_channel:
            x = self.expand_input_channels(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # self._qu_x_params = posterior
        return posterior

    def decode(self, z):
        if len(z.shape) == 5:
            orig_shape = z.shape
            z = self.post_quant_conv(z.view(-1, *orig_shape[2:]))
            dec = self.decoder(z)
            new_shape = [*orig_shape[:2], *dec.shape[1:]]
            dec = dec.view(*new_shape)
            if self.conform_to_3_channel:
                dec = dec.mean(dim=2, keepdim=True)
            if self.is_semseg and self.px_u == dist.Categorical:
                # Categorical distribution requires the category probabilities at the last dimension
                dec = dec.permute(0, 1, 3, 4, 2)

        else:
            z = self.post_quant_conv(z)
            dec = self.decoder(z)
            if self.conform_to_3_channel:
                dec = dec.mean(dim=1, keepdim=True)
            if self.is_semseg and self.px_u == dist.Categorical:
                # Categorical distribution requires the category probabilities at the last dimension
                dec = dec.permute(0, 2, 3, 1)

        if not self.return_lik_dist:
            return dec

        if self.is_semseg:
            # No scale for a Categorical or Bernoulli distribution
            dec_dist = self.px_u(dec)
        else:
            # Use a Fixed scale for the distribution as in MMVAE+ paper
            dec_dist = self.px_u(dec, torch.tensor(0.01).to(dec.device))
        return dec_dist

    def categorical_sample_to_one_hot(self, sample):
        if sample.dim() == 4:
            onehot = torch.zeros([*sample.shape[:2], self.ddconfig["in_channels"], *sample.shape[2:]]).to(sample.device)
            for i in range(self.ddconfig["in_channels"]):
                onehot[:, :, i] = sample == i
        elif sample.dim() == 3:
            onehot = torch.zeros([*sample.shape[0], self.ddconfig["in_channels"], *sample.shape[2:]]).to(sample.device)
            for i in range(self.ddconfig["in_channels"]):
                onehot[:, i] = sample == i

        return onehot

    def forward(self, input, sample_posterior=True, K=1):
        posterior = self.encode(input)
        z = self.sample(posterior, sample_posterior, K)
        dec = self.decode(z)
        return posterior, dec, z

    def sample(self, posterior, sample_posterior=True, K=1):
        if sample_posterior:
            z = posterior.sample(K, squeeze=False)
        else:
            z = posterior.mode()
            z = z.expand(K, *[-1 for _ in range(z.shape[1:])])

        return z


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

