# Objectives
import torch
from torch.profiler import record_function
from numpy import prod
from .objective_util import compute_microbatch_split, reconstruction_loss, register_objective, log_mean_exp, \
    MMDIndependenceLoss, CosSimIndependenceLoss
import pyiqa

eta = 1e-20


@register_objective("disentangling_objective")
class DisentanglingObjective(torch.nn.Module):
    def __init__(self,
                model,
                device,
                data_parallel=False,
                wlpips=0.0,
                wadversary=0.0,
                get_adversarial_loss_f=None,
                start_adv_iter=0,
                beta=1.0,
                channel_weighting=None,
                z_close_weight=50000,
                off_combination_weight=0.2,
                mmd_zw_weight=1,
                mmd_ww_weight=1,
                cos_zw_weight=1,
                cos_ww_weight=1,
                **kwargs):
        super(DisentanglingObjective, self).__init__()

        self.lpips = pyiqa.create_metric("lpips", device=device, as_loss=True)
        self.mmd_loss = MMDIndependenceLoss()
        self.mod = model.module.module if data_parallel else model.module
        self.all_combinations = self.mod.all_combinations

        self.get_adversarial_loss_f = get_adversarial_loss_f
        if wadversary > 0.0:
            assert self.get_adversarial_loss_f is not None

        self.wlpips = wlpips
        self.wadversary = wadversary
        self.start_adv_iter = start_adv_iter
        self.beta = beta
        self.channel_weighting = channel_weighting
        self.z_close_weight = z_close_weight
        self.off_combination_weight = off_combination_weight
        self.mmd_zw_weight = mmd_zw_weight
        self.mmd_ww_weight = mmd_ww_weight

        self.cos_zw_weight = cos_zw_weight
        self.cos_ww_weight = cos_ww_weight

    def _m_dreg(self, model, x, K=1, test=False, iteration=0):

        qu_xs, px_us, uss = model("self_and_cross_modal_generation_forward", x, K) if test else model("forward", x, K)

        qz_xs, qw_xs = [], []
        # for r, qu_x in enumerate(qu_xs_):
        for r, qu_x in enumerate(qu_xs):
            qu_x_r_mean, qu_x_r_std = qu_x.mean, qu_x.std  # mod.vaes[r].qu_x_params.mean, mod.vaes[r].qu_x_params.std

            assert not torch.isnan(qu_x_r_mean).any(), f"Model output is nan (qu_x_r_mean[{r}])"
            assert not torch.isnan(qu_x_r_std).any(), f"Model output is nan (qu_x_r_std[{r}])"

            qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [self.mod.w_channels, self.mod.z_channels], dim=1)
            qw_x_std, qz_x_std = torch.split(qu_x_r_std, [self.mod.w_channels, self.mod.z_channels], dim=1)
            # These detaches make me feel uncomfortable
            qw_x = self.mod.vaes[r].qu_x(qw_x_mean, qw_x_std)
            qz_x = self.mod.vaes[r].qu_x(qz_x_mean, qz_x_std)
            qz_xs.append(qz_x)
            qw_xs.append(qw_x)

        lws = []

        loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0, "z_dist": 0,
                     "mmd_zw": 0, "mmd_ww": 0, "z_dist_normed_euclidean": 0,
                     "discriminator_loss": 0, "adversarial_loss": 0,
                     "cos_sim_ww": 0, "cos_sim_zw": 0}

        # reconstruction loss
        for r, qu_x in enumerate(qu_xs):
            lpx_u = 0
            for d, px_u in enumerate(px_us[r]):
                wlpips_d = self.wlpips[d] if hasattr(self.wlpips, "__getitem__") else self.wlpips
                if self.all_combinations:
                    lpx_u = lpx_u + reconstruction_loss(px_u[0], x[d], self.mod.vaes[d].llik_scaling, loss_dict, wlpips_d, self.lpips,
                                                 self.channel_weighting[d], fail_on_nan=True)
                    lpx_u = lpx_u + reconstruction_loss(px_u[1], x[d], self.mod.vaes[d].llik_scaling, loss_dict, wlpips_d, self.lpips,
                                                 self.channel_weighting[d], fail_on_nan=True) * self.off_combination_weight
                else:
                    lpx_u = lpx_u + reconstruction_loss(px_u, x[d], self.mod.vaes[d].llik_scaling, loss_dict, wlpips_d, self.lpips,
                                                self.channel_weighting[d],
                                                fail_on_nan=True)

            lws.append(lpx_u)

        # Adversarial loss
        if self.wadversary > 0:
            for r, px_u in enumerate(px_us):
                if iteration > self.start_adv_iter:
                    al = self.get_adversarial_loss_f(px_u)
                    lws[r] = lws[r] - al * self.wadversary
                    loss_dict["adversarial_loss"] -= al.detach().cpu().sum().item() * self.wadversary

        # KL divergence loss for w
        for r, qw_x in enumerate(qw_xs):
            for param in self.mod.pw_params:
                assert not torch.isnan(param).any(), f"Model output is nan (pw_params[{r}])"
            pw = self.mod.pw(*self.mod.pw_params)
            """
            ws, zs = torch.split(uss[r], [self.mod.w_channels, self.mod.z_channels], dim=2)
            lpw = pw.log_prob(ws).sum(dim=(-1, -2, -3))
            lqw_x = qw_xs[r].log_prob(ws).sum(dim=(-1, -2, -3))
            kl_div = (lqw_x - lpw)
            """
            kl_div = torch.distributions.kl.kl_divergence(qw_x, pw).sum(dim=(-1, -2, -3)).unsqueeze(0)  # unsqueeze for K
            # print("kldiv w shape: ", kl_div.shape)
            lws[r] = lws[r] - kl_div * self.beta
            loss_dict["prior"] -= kl_div.detach().cpu().sum().item() * self.beta

        # KL divergence loss for z
        for r, qz_x in enumerate(qz_xs):
            for param in self.mod.pz_params:
                assert not torch.isnan(param).any(), f"Model output is nan (pz_params[{r}])"
            pz = self.mod.pz(*self.mod.pz_params)
            """
            ws, zs = torch.split(uss[r], [self.mod.w_channels, self.mod.z_channels], dim=2)
            lpz = pz.log_prob(zs).sum(dim=(-1, -2, -3))
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(dim=(-1, -2, -3)) for qz_x in qz_xs]))
            kl_div = (lqz_x - lpz)
            """
            kl_div = torch.distributions.kl.kl_divergence(qz_x, pz).sum(dim=(-1, -2, -3)).unsqueeze(0)  # unsqueeze for K
            # print("kldiv z shape: ", kl_div.shape)
            lws[r] = lws[r] - kl_div * self.beta
            loss_dict["prior"] -= kl_div.detach().cpu().sum().item() * self.beta


        # Z distance loss (Symmetric kl but only works for 2 modalities)
        if len(qu_xs) != 2:
            print("WARNING: Z distance loss only works for 2 modalities, skipping")
        else:
            """
            kl_div1 = torch.distributions.kl.kl_divergence(qz_xs[0], qz_xs[1]).sum(dim=(-1, -2, -3)).unsqueeze(0)
            kl_div2 = torch.distributions.kl.kl_divergence(qz_xs[1], qz_xs[0]).sum(dim=(-1, -2, -3)).unsqueeze(0)
            lws[0] = lws[0] - kl_div1 * self.z_close_weight
            lws[1] = lws[1] - kl_div2 * self.z_close_weight
            loss_dict["z_dist"] -= kl_div1.detach().cpu().sum().item()
            loss_dict["z_dist"] -= kl_div2.detach().cpu().sum().item()
            """

            z1_norm = torch.linalg.vector_norm(qz_xs[0].mean, dim=(1, 2, 3))
            z2_norm = torch.linalg.vector_norm(qz_xs[1].mean, dim=(1, 2, 3))
            z_dist_loss = torch.sum((qz_xs[0].mean - qz_xs[1].mean)**2, dim=(1, 2, 3))/(z1_norm * z2_norm)[:, None, None, None]

            lws[0] = lws[0] - z_dist_loss.unsqueeze(0) * self.z_close_weight/2
            lws[1] = lws[1] - z_dist_loss.unsqueeze(0) * self.z_close_weight/2
            loss_dict["z_dist_normed_euclidean"] = -z_dist_loss.detach().cpu().sum().item()


        # z - w independence loss with MMD
        mmd_loss = MMDIndependenceLoss()

        for r, (qz_x, qw_x_r) in enumerate(zip(qz_xs, qw_xs)):
            lmmd_zw_r = 0
            lmmd_ww_r = 0
            for d, qw_x in enumerate(qw_xs):
                lmmd_zw_r = lmmd_zw_r + mmd_loss(qz_x.mean, qw_x.mean)
                if r != d:
                    lmmd_ww_r = lmmd_ww_r + mmd_loss(qw_x_r.mean, qw_x.mean)

            loss_dict["mmd_zw"] -= lmmd_zw_r.detach().cpu().sum().item()
            loss_dict["mmd_ww"] -= lmmd_ww_r.detach().cpu().sum().item()

            lws[r] = lws[r] - lmmd_zw_r * self.mmd_ww_weight
            lws[r] = lws[r] - lmmd_ww_r * self.mmd_zw_weight

        # z - w cosine similarity loss
        cos_sim = CosSimIndependenceLoss()

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

        return torch.stack(lws), torch.stack(uss), loss_dict, (qu_xs, px_us, uss)

    def forward(self, model, x, K=1, test=False, **kwargs):
        S = x[0].shape[0]  # compute_microbatch_split(x, K)
        x_split = zip(*[_x.split(S) for _x in x])

        lw, uss, loss_dicts, model_outs = zip(*[self._m_dreg(model, _x, K, test, **kwargs) for _x in x_split])
        combined_loss_dict = {key: sum([loss_dict[key] for loss_dict in loss_dicts]) for key in loss_dicts[0].keys()}

        lw = torch.cat(lw, 2)  # concat on batch
        uss = torch.cat(uss, 2)  # concat on batch
        model_outs = model_outs[0]  # doing this concatenation is hard but I reckon ill always fit a batch in memory
        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
            if uss.requires_grad:
                uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return (grad_wt * lw).mean(0).sum(), combined_loss_dict, model_outs







