# Objectives
import torch
from numpy import prod
from ldm.data.dataset_CelebAMask import MMCelebA
import math

__OBJECTIVES__ = {}


def register_objective(name: str):
    def wrapper(fn):
        if __OBJECTIVES__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OBJECTIVES__[name] = fn
        return fn
    return wrapper


def get_objective(name: str):
    if __OBJECTIVES__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OBJECTIVES__[name]


# Helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def reconstruction_loss(px_u, x_d, llik_scale, loss_dict, wlpips_d, lpips, channel_weighting_d, fail_on_nan=True):
    lpx_u_elem = 0

    if isinstance(px_u, torch.distributions.Categorical):
        x_num = MMCelebA.one_hot_semseg_to_index(x_d)
        # Todo: implement channel weighting for Categorical
        recon_log_prob = px_u.log_prob(x_num).view(*px_u.batch_shape[:2], -1).mul(llik_scale).sum(dim=-1)
    else:
        log_probs = px_u.log_prob(x_d)
        if channel_weighting_d is not None:
            log_probs = log_probs * channel_weighting_d[None, None, :, None, None]
        recon_log_prob = log_probs.view(*px_u.batch_shape[:2], -1).mul(llik_scale).sum(dim=-1)

    loss_dict["recon_logprob"] += recon_log_prob.detach().cpu().sum().item()
    lpx_u_elem += recon_log_prob

    if fail_on_nan:
        if isinstance(px_u, torch.distributions.Bernoulli) or isinstance(px_u, torch.distributions.Categorical):
            assert not torch.isnan(px_u.probs).any(), f"Model output is nan (px_u.probs)"
        else:
            assert not torch.isnan(px_u.loc).any(), f"Model output is nan (px_u.mean)"
            assert not torch.isnan(px_u.scale).any(), f"Model output is nan (px_u.variance)"

        assert not torch.isnan(lpx_u_elem).any(), f"Model output is nan (lpx_u_elem)"

    if wlpips_d > 0.0:
        lpips_lpx_u_loss = torch.tensor([lpips(x_di, px_ui) for x_di, px_ui in zip(x_d.view(-1, *x_d.shape[1:]), px_u.mode.view(-1, *x_d.shape[1:]))], device=x_d.device).reshape(*px_u.batch_shape[:2])
        loss_dict["lpips"] -= lpips_lpx_u_loss.detach().cpu().sum().item()
        lpx_u_elem += - wlpips_d * lpips_lpx_u_loss

    if fail_on_nan:
        assert not torch.isnan(lpx_u_elem).any(), f"Model output is nan (lpips)"

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


def gauss_kl_div_loss(mu, sigma):
    pu = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    qv = torch.distributions.Normal(mu, sigma)
    kl_div = torch.distributions.kl.kl_divergence(qv, pu).sum(dim=(-1, -2, -3))

    return kl_div


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)