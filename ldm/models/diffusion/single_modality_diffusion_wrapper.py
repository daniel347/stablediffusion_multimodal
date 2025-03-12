import torch
from .ddpm import LatentDiffusion


class SingleModalityDiffusionWrapper(LatentDiffusion):

    def __init__(self, target_mode=0, **latent_diff_params):
        super(SingleModalityDiffusionWrapper, self).__init__(**latent_diff_params)
        self.target_mode = target_mode

        self.latent_w_dim = self.first_stage_model.latent_channel_w
        self.latent_z_dim = self.first_stage_model.latent_channel_z

        self.num_vaes = len(self.first_stage_model.vaes)

    def forward(self, x, c=None, *args, **kwargs):
        latents_split = torch.split(x, [*[self.latent_w_dim for d in range(self.num_vaes)], self.latent_z_dim], dim=1)
        x_chosen = torch.cat([latents_split[self.target_mode], latents_split[-1]], dim=1)
        return super(SingleModalityDiffusionWrapper, self).forward(x_chosen, c, *args, **kwargs)

    def sample(self, *args, **kwargs):
        sample_chosen = super(SingleModalityDiffusionWrapper, self).sample(*args, **kwargs)
        w_chosen = sample_chosen[:, :self.latent_w_dim]
        z_chosen = sample_chosen[:, -self.latent_z_dim:]

        W = []
        for i, vae in enumerate(self.first_stage_model.vaes):
            if i == self.target_mode:
                W.append(w_chosen)
            else:
                # We add 1 to the dimension and squeeze at end for compatibility with the K dimension during autoencoder training
                W.append(self.first_stage_model.sample_w(i, "pw_params", (1, *sample_chosen.shape), dev=sample_chosen.device).squeeze(0))

        print([w.shape for w in W])
        return torch.cat(W + [z_chosen], dim=1)

