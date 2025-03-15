# Objectives
import torch
from numpy import prod
from ..utils import log_mean_exp, is_multidata
from ..ldm.data.dataset_CelebAMask import MMCelebA
from .objective_util import compute_microbatch_split, reconstruction_loss, gauss_kl_div_loss, get_objective, register_objective

eta = 1e-20


def _m_dreg_non_disentangled(model,
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
            **kwargs):

    qu_xs, px_us, uss = model("forward", x, K)
    mod = model.module.module if data_parallel else model.module

    loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0}
    lw = 0
    for r, px_u in enumerate(px_us):
        wlpips_r = wlpips[r] if hasattr(wlpips, "__getitem__") else wlpips
        lpx_u = reconstruction_loss(px_u, x[r], mod.llik_scales[r], loss_dict, wlpips_r, lpips, channel_weighting[r],
                                         fail_on_nan=fail_on_nan)
        lw += lpx_u

    for r, qu_x in enumerate(qu_xs):
        l_kl_div = gauss_kl_div_loss(qu_x.mean, qu_x.std).unsqueeze(0)
        lw += beta * l_kl_div
        loss_dict["prior"] += beta * l_kl_div.detach().cpu().sum().item()

    return lw, uss, loss_dict


@register_objective("non_disentangled_objective")
def m_dreg_non_disentangled(model, x, K=1, test=False, **kwargs):
    """
    DReG objective
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])

    lw, uss, loss_dicts = zip(*[_m_dreg_non_disentangled(model, _x, K, test, **kwargs) for _x in x_split])
    combined_loss_dict = {key: sum([loss_dict[key] for loss_dict in loss_dicts]) for key in loss_dicts[0].keys()}

    lw = torch.cat(lw, 1)  # concat on batch
    uss = torch.cat([torch.cat(uss_e, 2) for uss_e in uss], 1)  # concat on latent channel, then batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if uss.requires_grad:
            uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)  # Because uss isnt returned, Im pretty sure this wont be used
    return (grad_wt * lw).mean(0).sum(), combined_loss_dict
