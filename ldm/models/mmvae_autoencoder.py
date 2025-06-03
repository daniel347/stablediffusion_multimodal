# Base MMVAEplus class definition
import torch
import torch.nn as nn
from ..util import instantiate_from_config, remove_module_prefix
from torch.distributions import Normal, Laplace
import torch.nn.functional as F
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ..data.dataset_CelebAMask import MMCelebA
from .z_combiner_net import ZCombinerNet

class MMVAEplus(nn.Module):
    """
    MMVAEplus class definition.
    """
    def __init__(self, prior_dist, latent_channel_z, latent_channel_w, cross_w_mode="sample", vae_params=None, ckpt_path=None, avg_z=False, all_combinations=False, stop_self_z_grad=False, z_net_params=None, scale_per_channel=False):
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

        if z_net_params is not None:
            self.z_comb_net = instantiate_from_config(z_net_params)

        total_latent_channels = self.w_channels * len(self.vaes) + self.z_channels
        if scale_per_channel:
            self.latent_mean = nn.Parameter(torch.zeros(total_latent_channels))
            self.latent_scale = nn.Parameter(torch.ones(total_latent_channels) * 1.0 / 0.18215)
        else:
            self.latent_mean = nn.Parameter(torch.tensor(0.0))
            self.latent_scale = nn.Parameter(torch.tensor(1.0 / 0.18215))

        self.cross_w_mode = cross_w_mode  # Options are "sample" (original MMVAE+), "constant", "zero"
        self.avg_z = avg_z

        self.all_combinations = all_combinations

        self.latent_channel_z = latent_channel_z
        self.latent_channel_w = latent_channel_w

        self.can_cross_generate = True
        self.can_sample = False
        self.factorised_latent_space = True

        self.stop_self_z_grad = stop_self_z_grad

        self.use_z_comb_network = False


        if ckpt_path is not None:
            print(f"Loading MMVAE checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            ckpt = remove_module_prefix(ckpt)
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            if len(missing) > 0:
                print(f"Missing Keys:\n {missing}")
            if len(unexpected) > 0:
                print(f"\nUnexpected Keys:\n {unexpected}")

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

    def self_and_cross_generation_choose_latent_dist(self, x, latent_dist_params, K=1, all_combinations=None, stop_self_z_grad=False):
        """
        Choose the latent distribution for self- and cross-generation.
        Args:
            x: Input
            K: Number of samples for reparameterization in latent space
            latent_dist_params: Parameters of latent distribution. One of pw_params or pw_params_aux
        """

        if all_combinations is None: all_combinations = self.all_combinations

        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            # qu_x, px_u, us = vae(x[m], K=K) # Get Encoding dist, Decoding dist, Latents for unimodal VAE m modality
            qu_x = vae.encode(x[m])
            us = vae.sample(qu_x, K=K)

            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            # px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix

        zs = [torch.split(us, [self.latent_channel_w, self.latent_channel_z], dim=2)[1] for us in uss]
        if self.avg_z:
            print("Averaging z")
            z_avg = torch.stack(zs, dim=0).mean(dim=0)



        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e == d: # Fill diagonal with self-reconstructions
                    if stop_self_z_grad:
                        w_e, z_e = torch.split(us, [self.latent_channel_w, self.latent_channel_z], dim=2)
                        us_z_grad_stop = torch.cat([w_e, z_e.clone().detach()])
                    else:
                        us_z_grad_stop = us

                    if self.avg_z:
                        w_e, _ = torch.split(us, [self.latent_channel_w, self.latent_channel_z], dim=2)
                        us_combined = torch.cat([w_e, z_avg], dim=2)
                        px_us[e][d] = vae.decode(us_combined)
                    else:
                        if all_combinations:
                            latent_w_sample = self.sample_w(d, latent_dist_params, us.size(), x[0].device)
                            us_unkown_self_w = torch.cat([latent_w_sample, zs[e]], dim=2)
                            px_us[e][d] = [vae.decode(us_z_grad_stop), vae.decode(us_unkown_self_w)]
                        else:
                            px_us[e][d] = vae.decode(us_z_grad_stop)

                else:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    latents_w = self.sample_w(d, latent_dist_params, us.size(), x[0].device)

                    # Combine shared and resampled private latents
                    if self.avg_z:
                        us_combined = torch.cat([latents_w, z_avg], dim=2)
                    else:
                        us_combined = torch.cat([latents_w, zs[e]], dim=2)

                    # Get cross-reconstruction likelihood
                    if all_combinations:
                        target_w_known = torch.split(uss[d], [self.latent_channel_w, self.latent_channel_z], dim=2)[0]
                        us_known_cross_w = torch.cat([target_w_known, zs[e]], dim=2)
                        px_us[e][d] = [vae.decode(us_combined), vae.decode(us_known_cross_w)]
                    else:
                        px_us[e][d] = vae.decode(us_combined)

        return qu_xs, px_us, uss

    def sample_w(self, vae_ind, latent_dist_params, us_shape, dev):
        vae = self.vaes[vae_ind]

        if latent_dist_params == "pw_params":
            pw_params = self.pw_params
        elif latent_dist_params == "pw_params_aux":
            pw_params = vae.pw_params_aux
        else:
            raise ValueError(f"Latent dist parameters {latent_dist_params} not supported")

        # Get shared latents from encoding modality e

        # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality m
        if self.cross_w_mode == "sample":
            pw = vae.pw(*pw_params)
            latents_w = pw.rsample([us_shape[1]]).transpose(0, 1).to(dev)
        elif self.cross_w_mode == "constant":
            latents_w = pw_params[0].unsqueeze(0).expand(*us_shape[:2], -1, -1, -1).to(dev)
        elif self.cross_w_mode == "zero":
            latents_w = torch.zeros(*us_shape[:2], pw_params[1].shape[1:]).to(dev)

        return latents_w

    def forward(self, x, K=1, all_combinations=None):
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
        return self.self_and_cross_generation_choose_latent_dist(x, "pw_params_aux", K=K, all_combinations=all_combinations)

    def self_and_cross_modal_generation_forward(self, data, K=1, all_combinations=None):
        """
        Test-time self- and cross-model generation forward function.
        Args:
            data: Input

        Returns:
            Unimodal encoding distribution, Matrix of self- and cross-modal reconstruction distrubutions, Latent embeddings

        """
        return self.self_and_cross_generation_choose_latent_dist(data, "pw_params", K=K, all_combinations=all_combinations)


    def self_and_cross_modal_generation(self, data, all_combinations=None):
        """
        Test-time self- and cross-reconstruction.
        Args:
            data: Input

        Returns:
            Matrix of self- and cross-modal reconstructions

        """
        with torch.no_grad():
            _, px_us, _ = self.self_and_cross_modal_generation_forward(data, all_combinations=all_combinations)
            # ------------------------------------------------
            # cross-modal matrix of reconstructions
            recons = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

            for i in range(len(px_us)):
                for j in range(len(px_us[i])):
                    if isinstance(px_us[i][j], torch.distributions.Categorical):
                        recons[i][j] = MMCelebA.index_semseg_to_one_hot(px_us[i][j].mode)
                    elif isinstance(px_us[i][j], torch.distributions.Bernoulli):
                        recons[i][j] = px_us[i][j].mode
                    else:
                        recons[i][j] = px_us[i][j].mean

        return recons

    def forward_z_comb_net(self, data, sample=False):
        *w_dists, z_comb_dist = self.encode(data, return_dist=True, use_z_combnet=True)

        if sample:
            z = z_comb_dist.sample(squeeze=False)
            W = [dist.sample(squeeze=False) for dist in w_dists]
        else:
            z = z_comb_dist.mean.unsqueeze(0)
            W = [dist.mean.unsqueeze(0) for dist in w_dists]

        recons = self.decode(torch.cat([*W, z], dim=2), use_kdim=True)

        return [*w_dists, z_comb_dist], recons, [*W, z]

    def encode(self, data, avg_z=None, return_dist=False, use_z_combnet=None):
        if avg_z is None: avg_z = self.avg_z
        if use_z_combnet is None: use_z_combnet = self.use_z_comb_network

        # Encode data into z and W
        zs = []
        W = []

        qz_xs = []
        qw_xs = []

        for dm, vae in zip(data, self.vaes):
            qu_x = vae.encode(dm)
            latents_w, latents_z = torch.split(qu_x.mean, [self.latent_channel_w, self.latent_channel_z], dim=1)
            zs.append(latents_z)
            W.append(latents_w)

            if return_dist or use_z_combnet:
                logvar_w, logvar_z = torch.split(qu_x.logvar, [self.latent_channel_w, self.latent_channel_z], dim=1)
                qz_xs.append(DiagonalGaussianDistribution(torch.cat([latents_z, logvar_z], dim=1)))
                qw_xs.append(DiagonalGaussianDistribution(torch.cat([latents_w, logvar_w], dim=1)))

        if avg_z:
            z = torch.stack(zs).mean(dim=0)
            full_latent = torch.concat([*W, z], dim=1)
            return full_latent

        if use_z_combnet:
            z_comb_dist = self.z_comb_net(qz_xs)
            if return_dist:
                return [*qw_xs, z_comb_dist]
            else:
                return [*W, z_comb_dist.mean]

        else:
            if return_dist:
                return [*qw_xs, *qz_xs]
            else:
                return [*W, *zs]

    def decode(self, full_latent, use_kdim=False):
        # convert z and W to image and depth
        concat_dim = 2 if use_kdim else 1
        zW = torch.split(full_latent, [*[self.latent_channel_w for _ in range(len(self.vaes))], self.latent_channel_z], dim=concat_dim)
        z = zW[-1]
        W = zW[:-1]

        recons = []
        for vae, w in zip(self.vaes, W):
            recons.append(vae.decode(torch.cat([w, z], dim=concat_dim)))

        return recons

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
