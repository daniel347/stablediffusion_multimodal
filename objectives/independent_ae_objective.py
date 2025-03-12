# Objectives
import torch
from numpy import prod
from ..utils import log_mean_exp, is_multidata
from ..ldm.data.dataset_CelebAMask import MMCelebA

eta = 1e-20

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
            **kwargs):

    qu_xs, px_us, uss = model("forward", x, K)
    mod = model.module.module if data_parallel else model.module

    qz_xs, qw_xs = [], []
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

    loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0}
    for r, qu_x in enumerate(qu_xs):
        if fail_on_nan:
            assert not torch.isnan(uss[r]).any(), f"Model output is nan (uss[{r}])"
            for param in mod.pu_params:
                assert not torch.isnan(param).any(), f"Model output is nan (pz_params[{r}])"

        lpu = mod.pz(*mod.pz_params).log_prob(uss[r]).sum(dim=(-1, -2, -3))
        lqu_x = log_mean_exp(qu_x.log_prob(uss[r]).sum(dim=(-1, -2, -3)))

        wlpips_r = wlpips[r] if hasattr(wlpips, "__getitem__") else wlpips
        lpx_u = []
        lpx_u_elem = reconstruction_loss(px_us[r][0], x[r], mod.vaes[r], loss_dict, wlpips_r, lpips, channel_weighting[d],
                                         fail_on_nan=True)
        lpx_u.append(lpx_u_elem)

        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + beta * (lpu - lqu_x)

        loss_dict["prior"] += beta * (lpu - lqu_x).detach().cpu().sum().item()

        lws.append(lw)

    if train_adversary_f is not None:
        disc_loss, autoencoder_loss = train_adversary_f(x, px_us)
        lws = torch.stack(lws) - wadversary * autoencoder_loss / len(qu_xs)
        return lws, torch.stack(uss)

    return torch.stack(lws), torch.stack(uss), loss_dict


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


def train_adversary(real_data, px_us, discriminator, optimiser, device):
    optimiser.zero_grad()

    real_output = torch.stack(discriminator(real_data), dim=1)
    fake_outputs_detach = [torch.stack(discriminator([t.detach() for t in px_u]), dim=1) for px_u in px_us]

    real_labels = torch.ones(real_output.shape).to(device)
    fake_labels = torch.zeros(fake_outputs_detach[0].shape).to(device)

    weight_real = 1.0
    weight_fake = 1.0

    real_loss = torch.nn.functional.binary_cross_entropy(real_output, real_labels) * weight_real
    fake_loss = 0
    for out in fake_outputs_detach:
        fake_loss += torch.nn.functional.binary_cross_entropy(out, fake_labels) * weight_fake

    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    optimiser.step()

    fake_outputs = [torch.stack(discriminator(px_u), dim=1) for px_u in px_us]
    autoencoder_loss = 0
    for out in fake_outputs:
        autoencoder_loss += torch.nn.functional.binary_cross_entropy(out, real_labels) * weight_fake

    return disc_loss, autoencoder_loss


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
