import torch
from numpy import prod
from .objective_util import compute_microbatch_split, reconstruction_loss, register_objective, log_mean_exp, MMDIndependenceLoss
import pyiqa

eta = 1e-20


@register_objective("z_combiner_objective_4way")
class ZCombinerObj4Way(torch.nn.Module):
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
                off_combination_weight=0.2,
                mmd_zw_weight=1,
                mmd_ww_weight=1,
                **kwargs):
        super(ZCombinerObj4Way, self).__init__()

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
        self.off_combination_weight = off_combination_weight
        self.mmd_zw_weight = mmd_zw_weight
        self.mmd_ww_weight = mmd_ww_weight

    def _m_dreg(self, model, x, K=1, test=False, iteration=0, **kwargs):

        posteriors, recons, us = model("forward_z_comb_4way", x, sample=not test, test=test)
        *qw_xs, qz_x = posteriors

        qw_xs = [torch.distributions.Normal(qw_x.mean, qw_x.std) for qw_x in qw_xs]
        qz_x = torch.distributions.Normal(qz_x.mean, qz_x.std)

        loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0, "mmd_zw": 0, "adversarial_loss": 0}

        lws = []

        # reconstruction loss
        lpx_u = 0
        for r in range(len(recons)):
            lpx_u = 0
            for d, px_u in enumerate(recons[r]):
                wlpips_d = self.wlpips[d] if hasattr(self.wlpips, "__getitem__") else self.wlpips
                wll_d = self.wll[d] if hasattr(self.wll, "__getitem__") else self.wll
                lpx_u = lpx_u + reconstruction_loss(px_u, x[d], self.mod.vaes[d].llik_scaling, loss_dict, wlpips_d, wll_d, self.lpips,
                                                    self.channel_weighting[d],
                                                    fail_on_nan=True)

            lws.append(lpx_u)

        # Adversarial loss
        if self.wadversary > 0:
            for r, px_u in enumerate(recons):
                if iteration > self.start_adv_iter:
                    al = self.get_adversarial_loss_f(px_u)
                    lws[r] = lws[r] - al * self.wadversary
                    loss_dict["adversarial_loss"] += al.detach().cpu().sum().item() * self.wadversary

        # print(f"lws len: {len(lws)}  lws shapes: {[l.shape for l in lws]}")

        # KL divergence loss for z
        for param in self.mod.pz_params:
            assert not torch.isnan(param).any(), f"Model output is nan (pz_params[{r}])"
        pz = self.mod.pz(*self.mod.pz_params)
        kl_div = torch.distributions.kl.kl_divergence(qz_x, pz).sum(dim=(-1, -2, -3)).unsqueeze(0)  # unsqueeze for K
        # print("kldiv z shape: ", kl_div.shape)
        lws = [lws_val - kl_div * self.beta for lws_val in lws]
        loss_dict["prior"] -= kl_div.detach().cpu().sum().item() * self.beta


        # z - w independence loss with MMD
        mmd_loss = MMDIndependenceLoss(target_dim=1)

        for r, qw_x_r in enumerate(qw_xs):
            print(f"qz_x shape: {qz_x.mean.shape}   qw_x_r shape: {qw_x_r.mean.shape}")
            lmmd_zw_r = mmd_loss(qz_x.mean, qw_x_r.mean)
            loss_dict["mmd_zw"] -= lmmd_zw_r.detach().cpu().sum().item()
            lws[0] = lws[0] - lmmd_zw_r * self.mmd_ww_weight

        return torch.stack(lws), torch.cat(us, dim=2), loss_dict, {"U_dist": posteriors, "Xhat": recons, "U_samp": us}

    def forward(self, model, x, K=1, test=False, **kwargs):
        S = x[0].shape[0] # compute_microbatch_split(x, K)
        x_split = zip(*[_x.split(S) for _x in x])

        lw, uss, loss_dicts, model_outs = zip(*[self._m_dreg(model, _x, K, test, **kwargs) for _x in x_split])
        combined_loss_dict = {key: sum([loss_dict[key] for loss_dict in loss_dicts]) for key in loss_dicts[0].keys()}
        print(f"lw len: {len(lw)} shapes: {[l.shape for l in lw]}, uss shapes: {[u.shape for u in uss]}")

        lw = torch.cat(lw, 2)  # concat on batch
        uss = torch.cat(uss, 1)  # concat on batch
        model_outs = model_outs[0]
        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
            if uss.requires_grad:
                uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return (grad_wt * lw).mean(0).sum(), combined_loss_dict, model_outs