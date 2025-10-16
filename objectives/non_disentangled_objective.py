# Objectives
import torch
from numpy import prod
from .objective_util import compute_microbatch_split, reconstruction_loss, gauss_kl_div_loss, get_objective, register_objective, log_mean_exp, is_multidata
import pyiqa

eta = 1e-20

@register_objective("non_disentangling_objective")
class NonDisentanglingObjective(torch.nn.Module):
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
                **kwargs):
        super(NonDisentanglingObjective, self).__init__()

        self.lpips = pyiqa.create_metric("lpips", device=device, as_loss=True)
        self.mod = model.module.module if data_parallel else model.module

        self.get_adversarial_loss_f = get_adversarial_loss_f
        if wadversary > 0.0:
            assert self.get_adversarial_loss_f is not None

        self.wlpips = wlpips
        self.wll = wll
        self.wadversary = wadversary
        self.start_adv_iter = start_adv_iter
        self.beta = beta
        self.channel_weighting = channel_weighting

    def _m_dreg(self, model, x, K=1, test=False, iteration=0, **kwargs):
        qu_xs, px_us, uss = model("forward", x, K)

        loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0, "adversarial_loss": 0}
        lw = 0
        for r, px_u in enumerate(px_us):
            if isinstance(px_u, list):
                px_u = px_u[r]   # if we return a 2D list just take the diagonal
            wlpips_r = self.wlpips[r] if hasattr(self.wlpips, "__getitem__") else self.wlpips
            wll_r = self.wll[r] if hasattr(self.wll, "__getitem__") else self.wll
            lpx_u = reconstruction_loss(px_u, x[r], self.mod.llik_scales[r], loss_dict, wlpips_r, wll_r, self.lpips, self.channel_weighting[r])
            lw = lw + lpx_u

        # KL divergence loss
        for r, qu_x in enumerate(qu_xs):
            l_kl_div = gauss_kl_div_loss(qu_x.mean, qu_x.std).unsqueeze(0)
            lw = lw - self.beta * l_kl_div
            loss_dict["prior"] -= l_kl_div.detach().cpu().sum().item()

        # Adversarial loss
        if self.wadversary > 0:
            if iteration > self.start_adv_iter:
                al = self.get_adversarial_loss_f(px_us)
                lw = lw - al * self.wadversary
                loss_dict["adversarial_loss"] -= al.detach().cpu().sum().item() * self.wadversary

        return lw.unsqueeze(0), torch.stack(uss), loss_dict, (qu_xs, px_us, uss)

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