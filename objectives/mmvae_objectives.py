# Objectives
import torch
from numpy import prod
from ..utils import log_mean_exp, is_multidata
from ..ldm.data.dataset_CelebAMask import MMCelebA
from .objective_util import compute_microbatch_split, reconstruction_loss, gauss_kl_div_loss, get_objective, register_objective

eta = 1e-20

# MULTIMODAL OBJECTIVES
def _m_elbo(model, x, K=1, test=False):
    if test:
        qu_xs, px_us, uss = model.self_and_cross_modal_generation_forward(x, K)
    else:
        qu_xs, px_us, uss = model(x, K)
    qz_xs, qw_xs = [], []
    for r, qu_x in enumerate(qu_xs):
        qu_x_r_mean, qu_x_r_lv = model.vaes[r].qu_x_params.mean, model.vaes[r].qu_x_params.logvar
        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [model.w_channels, model.z_channels], dim=1)
        qw_x_lv, qz_x_lv = torch.split(qu_x_r_lv, [model.w_channels, model.z_channels], dim=1)
        qw_x = model.vaes[r].qu_x(qw_x_mean, qw_x_lv)
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    lws = []
    for r, qu_x in enumerate(qu_xs):
        ws, zs = torch.split(uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=1)
        lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
        lpw = model.pw(*model.pw_params).log_prob(ws).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(-1)
        lpx_u = [px_u.log_prob(x[d]).view(*px_u.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_u in enumerate(px_us[r])]

        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + model.params.beta*(lpz + lpw - lqz_x - lqw_x)
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_elbo(model, x, K=1, test=False):
    """
    ELBO (IWAE) objective
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_elbo(model, _x, K, test) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _m_dreg(model,
            x,
            K=1,
            test=False,
            data_parallel=False,
            wlpips=0.0,
            wadversary=0.0,
            train_adversary_f=None,
            fail_on_nan=True,
            beta=1.0,
            lpips=None,
            channel_weighting=None,
            z_close_weight=0.1,
            off_combination_weight=0.2,
            **kwargs):

    if test:
        qu_xs, px_us, uss = model("self_and_cross_modal_generation_forward", x, K)
    else:
        qu_xs, px_us, uss = model("forward", x, K)
    mod = model.module.module if data_parallel else model.module
    all_combinations = mod.all_combinations

    qz_xs, qw_xs = [], []
    # for r, qu_x in enumerate(qu_xs_):
    for r, qu_x in enumerate(qu_xs):
        qu_x_r_mean, qu_x_r_std = qu_x.mean, qu_x.std  # mod.vaes[r].qu_x_params.mean, mod.vaes[r].qu_x_params.std
        if fail_on_nan:
            assert not torch.isnan(qu_x_r_mean).any(), f"Model output is nan (qu_x_r_mean[{r}])"
            assert not torch.isnan(qu_x_r_std).any(), f"Model output is nan (qu_x_r_std[{r}])"

        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [mod.w_channels, mod.z_channels], dim=1)
        qw_x_std, qz_x_std = torch.split(qu_x_r_std, [mod.w_channels, mod.z_channels], dim=1)
        qw_x = mod.vaes[r].qu_x(qw_x_mean.detach(), qw_x_std.detach())
        qz_x = mod.vaes[r].qu_x(qz_x_mean.detach(), qz_x_std.detach())
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    lws = []

    zs = [torch.split(us, [mod.w_channels, mod.z_channels], dim=2)[1] for us in uss]
    z_avg = torch.stack(zs, dim=0).mean(dim=0)

    loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0, "z_dist": 0}
    for r, qu_x in enumerate(qu_xs):
        ws, zs = torch.split(uss[r], [mod.w_channels, mod.z_channels], dim=2)
        lz_dist = torch.linalg.vector_norm(zs - z_avg, dim=(-1, -2, -3))

        loss_dict["z_dist"] -= lz_dist.detach().cpu().sum().item()

        if fail_on_nan:
            assert not torch.isnan(uss[r]).any(), f"Model output is nan (uss[{r}])"

            for param in mod.pz_params:
                assert not torch.isnan(param).any(), f"Model output is nan (pz_params[{r}])"
            for param in mod.pw_params:
                assert not torch.isnan(param).any(), f"Model output is nan (pw_params[{r}])"

        lpz = mod.pz(*mod.pz_params).log_prob(zs).sum(dim=(-1, -2, -3))
        lpw = mod.pw(*mod.pw_params).log_prob(ws).sum(dim=(-1, -2, -3))
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(dim=(-1, -2, -3)) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(dim=(-1, -2, -3))

        lpx_u = []
        for d, px_u in enumerate(px_us[r]):
            wlpips_d = wlpips[d] if hasattr(wlpips, "__getitem__") else wlpips
            if all_combinations:
                lpx_u_elem = reconstruction_loss(px_u[0], x[d], mod.vaes[d], loss_dict, wlpips_d, lpips, channel_weighting[d], fail_on_nan=True)
                lpx_u_elem += reconstruction_loss(px_u[1], x[d], mod.vaes[d], loss_dict, wlpips_d, lpips, channel_weighting[d], fail_on_nan=True) * off_combination_weight
            else:
                lpx_u_elem = reconstruction_loss(px_u, x[d], mod.vaes[d], loss_dict, wlpips_d, lpips, channel_weighting[d], fail_on_nan=True)
            lpx_u.append(lpx_u_elem)

        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + beta * (lpz + lpw - lqz_x - lqw_x) - z_close_weight * lz_dist

        loss_dict["prior"] += beta * (lpz + lpw - lqz_x - lqw_x).detach().cpu().sum().item()

        lws.append(lw)

    if train_adversary_f is not None:
        disc_loss, autoencoder_loss = train_adversary_f(x, px_us)
        lws = torch.stack(lws) - wadversary * autoencoder_loss / len(qu_xs)
        return lws, torch.stack(uss)

    return torch.stack(lws), torch.stack(uss), loss_dict


@register_objective("disentangled_objective")
def m_dreg(model, x, K=1, test=False, **kwargs):
    """
    DReG objective
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])

    lw, uss, loss_dicts = zip(*[_m_dreg(model, _x, K, test, **kwargs) for _x in x_split])
    combined_loss_dict = {key: sum([loss_dict[key] for loss_dict in loss_dicts]) for key in loss_dicts[0].keys()}

    lw = torch.cat(lw, 2)  # concat on batch
    uss = torch.cat(uss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if uss.requires_grad:
            uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum(), combined_loss_dict


