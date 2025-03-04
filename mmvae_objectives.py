# Objectives
import torch
from numpy import prod
from utils import log_mean_exp, is_multidata
from ldm.data.dataset_CelebAMask import MMCelebA

eta = 1e-20

# Helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

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


def _m_dreg(model, x, K=1, test=False, data_parallel=False, wlpips=0.0, wadversary=0.0, train_adversary_f=None, fail_on_nan=True, **kwargs):
    beta = kwargs.get("beta", 1.0)

    lpips = kwargs.get("lpips", None)
    # Use a channel weighting to weight classes in semantic segmentation
    channel_weighting = kwargs.get("channel_weighting", None)

    if test:
        qu_xs, px_us, uss = model("self_and_cross_modal_generation_forward", x, K)
    else:
        qu_xs, px_us, uss = model("forward", x, K)
    mod = model.module.module if data_parallel else model.module
    qu_xs_ = [vae.qu_x_params.detach() for vae in mod.vaes]
    qz_xs, qw_xs = [], []
    # for r, qu_x in enumerate(qu_xs_):
    for r, qu_x in enumerate(qu_xs_):
        qu_x_r_mean, qu_x_r_std = mod.vaes[r].qu_x_params.mean, mod.vaes[r].qu_x_params.std
        if fail_on_nan:
            assert not torch.isnan(qu_x_r_mean).any(), f"Model output is nan (qu_x_r_mean[{r}])"
            assert not torch.isnan(qu_x_r_std).any(), f"Model output is nan (qu_x_r_std[{r}])"

        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [mod.w_channels, mod.z_channels], dim=1)
        qw_x_std, qz_x_std = torch.split(qu_x_r_std, [mod.w_channels, mod.z_channels], dim=1)
        qw_x = mod.vaes[r].qu_x(qw_x_mean, qw_x_std)
        qz_x = mod.vaes[r].qu_x(qz_x_mean, qz_x_std)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    lws = []
    for r, qu_x in enumerate(qu_xs_):
        ws, zs = torch.split(uss[r], [mod.w_channels, mod.z_channels], dim=2)

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
            lpx_u_elem = 0

            if isinstance(px_u, torch.distributions.Categorical):
                x_num = MMCelebA.one_hot_semseg_to_index(x[d])
                # Todo: implement channel weighting for Categorical
                lpx_u_elem += px_u.log_prob(x_num).view(*px_u.batch_shape[:2], -1).mul(mod.vaes[d].llik_scaling).sum(dim=(-1, -2, -3))
            else:
                log_probs = px_u.log_prob(x[d])
                if channel_weighting[d] is not None:
                    log_probs = log_probs * channel_weighting[d][None, None, :, None, None]
                lpx_u_elem += log_probs.view(*px_u.batch_shape[:2], -1).mul(mod.vaes[d].llik_scaling).sum(dim=(-1, -2, -3))

            if fail_on_nan:
                if isinstance(px_u, torch.distributions.Bernoulli) or isinstance(px_u, torch.distributions.Categorical):
                    assert not torch.isnan(px_u.probs).any(), f"Model output is nan (px_u.probs[{r}][{d}])"
                else:
                    assert not torch.isnan(px_u.loc).any(), f"Model output is nan (px_u.mean[{r}][{d}])"
                    assert not torch.isnan(px_u.scale).any(), f"Model output is nan (px_u.variance[{r}][{d}])"

                assert not torch.isnan(lpx_u_elem).any(), f"Model output is nan (lpx_u_elem[{r}][{d}])"

            wlpips_d = wlpips[d] if hasattr(wlpips, "__getitem__") else wlpips
            if wlpips_d > 0.0:
                lpx_u_elem += - wlpips_d * lpips(x[d].squeeze(0), px_u.mode.squeeze(0))

            if fail_on_nan:
                assert not torch.isnan(lpx_u_elem).any(), f"Model output is nan (lpips [{r}][{d}])"

            lpx_u.append(lpx_u_elem)


        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + beta * (lpz + lpw - lqz_x - lqw_x)

        lws.append(lw)

    if train_adversary_f is not None:
        disc_loss, autoencoder_loss = train_adversary_f(x, px_us[r])
        lws = torch.stack(lws) - wadversary * autoencoder_loss / len(qu_xs_)
        return lws, torch.stack(uss)

    return torch.stack(lws), torch.stack(uss)


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
    lw, uss = zip(*[_m_dreg(model, _x, K, test, **kwargs) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    uss = torch.cat(uss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if uss.requires_grad:
            uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum()


