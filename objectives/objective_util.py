# Objectives
import torch
from numpy import prod
from ..utils import log_mean_exp, is_multidata
from ..ldm.data.dataset_CelebAMask import MMCelebA

# Helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def reconstruction_loss(px_u, x_d, mod_vae, loss_dict, wlpips_d, lpips, channel_weighting_d, fail_on_nan=True):
    lpx_u_elem = 0

    if isinstance(px_u, torch.distributions.Categorical):
        x_num = MMCelebA.one_hot_semseg_to_index(x_d)
        # Todo: implement channel weighting for Categorical
        recon_log_prob = px_u.log_prob(x_num).view(*px_u.batch_shape[:2], -1).mul(mod_vae.llik_scaling).sum(dim=(-1, -2, -3))
    else:
        log_probs = px_u.log_prob(x_d)
        if channel_weighting_d is not None:
            log_probs = log_probs * channel_weighting_d[None, None, :, None, None]
        recon_log_prob = log_probs.view(*px_u.batch_shape[:2], -1).mul(mod_vae.llik_scaling).sum(dim=(-1, -2, -3))

    loss_dict["recon_logprob"] += recon_log_prob.detach().cpu().sum().item()
    lpx_u_elem += recon_log_prob

    if fail_on_nan:
        if isinstance(px_u, torch.distributions.Bernoulli) or isinstance(px_u, torch.distributions.Categorical):
            assert not torch.isnan(px_u.probs).any(), f"Model output is nan (px_u.probs[{r}][{d}])"
        else:
            assert not torch.isnan(px_u.loc).any(), f"Model output is nan (px_u.mean[{r}][{d}])"
            assert not torch.isnan(px_u.scale).any(), f"Model output is nan (px_u.variance[{r}][{d}])"

        assert not torch.isnan(lpx_u_elem).any(), f"Model output is nan (lpx_u_elem[{r}][{d}])"

    if wlpips_d > 0.0:
        lpips_lpx_u_loss = lpips(x_d.squeeze(0), px_u.mode.squeeze(0))
        loss_dict["lpips"] -= lpips_lpx_u_loss.detach().cpu().sum().item()
        lpx_u_elem += - wlpips_d * lpips_lpx_u_loss

    if fail_on_nan:
        assert not torch.isnan(lpx_u_elem).any(), f"Model output is nan (lpips [{r}][{d}])"

    return lpx_u_elem
