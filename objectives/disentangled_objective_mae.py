# Objectives
import torch
from torch.profiler import record_function
from numpy import prod
from .objective_util import compute_microbatch_split, reconstruction_loss, register_objective, log_mean_exp, \
    MMDIndependenceLoss, CosSimIndependenceLoss
import pyiqa

eta = 1e-20


@register_objective("disentangling_objective_mae")
class DisentanglingObjectiveMAE(torch.nn.Module):
    def __init__(self,
                model,
                device,
                data_parallel=False,
                wlpips=0.0,
                wll=1.0,
                wadversary=0.0,
                get_adversarial_loss_f=None,
                start_adv_iter=0,
                beta=1.0,
                channel_weighting=None,
                z_close_weight=50000,
                mmd_zw_weight=1,
                mmd_ww_weight=1,
                **kwargs):
        super(DisentanglingObjectiveMAE, self).__init__()

        self.lpips = pyiqa.create_metric("lpips", device=device, as_loss=True)
        self.mmd_loss = MMDIndependenceLoss()
        self.mod = model.module.module if data_parallel else model.module
        self.all_combinations = self.mod.all_combinations

        self.get_adversarial_loss_f = get_adversarial_loss_f
        if wadversary > 0.0:
            assert self.get_adversarial_loss_f is not None

        self.wlpips = wlpips
        self.wll = wll
        self.wadversary = wadversary
        self.start_adv_iter = start_adv_iter
        self.beta = beta
        self.channel_weighting = channel_weighting
        self.z_close_weight = z_close_weight
        self.mmd_zw_weight = mmd_zw_weight
        self.mmd_ww_weight = mmd_ww_weight

        self.weights = torch.tensor([self.wll,
                                    self.wlpips,
                                    self.wadversary,
                                    self.beta * z_close_weight,
                                    self.beta * self.mmd_zw_weight,
                                    self.beta * self.mmd_ww_weight])

    def _m_dreg(self, model, X, iteration=0, model_params={}, **kwargs):
        loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0, "z_dist": 0,
                     "mmd_zw": 0, "mmd_ww": 0, "z_dist_normed_euclidean": 0,
                     "discriminator_loss": 0, "adversarial_loss": 0,
                     "cos_sim_ww": 0, "cos_sim_zw": 0}

        W, Z, z_comb, Xhat = model(X, **model_params)
        px_us = [self.mod.lik_dists[d](Xhat[d], **self.mod.lik_dist_params[d]) for d in range(X.shape[1])]

        lws = []

        # reconstruction loss
        lpx_u = 0
        for d, xhat in enumerate(Xhat):
            wlpips_d = self.wlpips[d] if hasattr(self.wlpips, "__getitem__") else self.wlpips
            wll_d = self.wll[d] if hasattr(self.wll, "__getitem__") else self.wll

            px_u = self.mod.lik_dists[d](xhat, **self.mod.lik_dist_params[d])
            # TODO: what about the stupid k dimension
            lpx_u = lpx_u + reconstruction_loss(px_u, X[d], self.mod.vaes[d].llik_scaling, loss_dict,
                                                1 if wlpips_d > 0 else 0, 1, self.lpips,
                                                self.channel_weighting[d],
                                                fail_on_nan=False, return_breakdown=True)
        for a in range(lpx_u.shape[-1]):
            lws.append(lpx_u[:, :, a])

        K, B = lws[0].shape

        # Adversarial loss
        if self.wadversary > 0:
            for r, px_u in enumerate(px_us):
                if iteration > self.start_adv_iter:
                    al = self.get_adversarial_loss_f(px_u)
                    lws[r].append(-al)
                    loss_dict["adversarial_loss"] -= al.clone().detach().cpu().sum().item() * self.wadversary
                else:
                    lws[r].append(torch.zeros(K, B, dtype=torch.float32, device=px_u[0].device))

        # TODO: vector quantisation loss?

        # Z distance loss (Symmetric kl but only works for 2 modalities)
        if len(Z) != 2:
            print("WARNING: Z distance loss only works for 2 modalities, skipping")
        else:
            z1_norm = torch.linalg.vector_norm(Z[0], dim=(1, 2))
            z2_norm = torch.linalg.vector_norm(Z[1], dim=(1, 2))
            z_dist_loss = torch.sum((Z[0] - Z[1].mean)**2, dim=(1, 2))/(z1_norm * z2_norm)

            lws.append(-z_dist_loss.unsqueeze(0)/2)
            loss_dict["z_dist_normed_euclidean"] = -z_dist_loss.clone().detach().cpu().sum().item()


        # z - w independence loss with MMD
        mmd_loss = MMDIndependenceLoss(target_dim=2)

        for r, (qz_x, qw_x_r) in enumerate(zip(Z, W)):
            lmmd_zw_r = 0
            lmmd_ww_r = 0
            for d, qw_x in enumerate(W):
                print(f"qz_x mean: {qz_x.mean.shape}")
                print(f"qw_x mean: {qw_x.mean.shape}")
                lmmd_zw_r = lmmd_zw_r + mmd_loss(qz_x, qw_x)
                if r != d:
                    lmmd_ww_r = lmmd_ww_r + mmd_loss(qw_x_r, qw_x)

            lws[r].append( - lmmd_zw_r * torch.ones(K, B, dtype=torch.float32, device=lmmd_zw_r.device))
            lws[r].append( - lmmd_ww_r * torch.ones(K, B, dtype=torch.float32, device=lmmd_zw_r.device))

            loss_dict["mmd_zw"] -= lmmd_zw_r.clone().detach().cpu().sum().item()
            loss_dict["mmd_ww"] -= lmmd_ww_r.clone().detach().cpu().sum().item()

        """        # z - w cosine similarity loss
        cos_sim = CosSimIndependenceLoss(target_dim=1)

        for r, (qz_x, qw_x_r) in enumerate(zip(qz_xs, qw_xs)):
            lcos_zw_r = 0
            lcos_ww_r = 0
            for d, qw_x in enumerate(qw_xs):
                lcos_zw_r = lcos_zw_r + cos_sim(qz_x.mean, qw_x.mean)
                if r != d:
                    lcos_ww_r = lcos_ww_r + cos_sim(qw_x_r.mean, qw_x.mean)

            loss_dict["cos_sim_zw"] -= lcos_zw_r.detach().cpu().sum().item()
            loss_dict["cos_sim_ww"] -= lcos_ww_r.detach().cpu().sum().item()

            lws[r] = lws[r] - lcos_zw_r * self.cos_ww_weight
            lws[r] = lws[r] - lcos_ww_r * self.cos_zw_weight
        """

        # stack loss type
        lws = [torch.stack(lw, -1) for lw in lws]

        return torch.stack(lws), loss_dict, (W, Z, z_comb, X)


    def forward(self, model, x, K=1, test=False, set_l0=False, **kwargs):
        S = x[0].shape[0]  # compute_microbatch_split(x, K)
        x_split = zip(*[_x.split(S) for _x in x])

        lw, loss_dicts, model_outs = zip(*[self._m_dreg(model, _x, K, test, **kwargs) for _x in x_split])
        combined_loss_dict = {key: sum([loss_dict[key] for loss_dict in loss_dicts]) for key in loss_dicts[0].keys()}

        lw = torch.cat(lw, 2)  # concat on batch
        model_outs = model_outs[0]  # doing this concatenation is hard but I reckon ill always fit a batch in memory
        print(f"loss shape: {lw.shape}")

        lw = lw.mean(dim=0).sum(dim=[0, 1])
        weighted_lw = lw @ self.weights

        return weighted_lw, combined_loss_dict, model_outs







