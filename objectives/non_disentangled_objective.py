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
                wadversary=0.0,
                beta=1.0,
                channel_weighting=None,
                **kwargs):
        super(NonDisentanglingObjective, self).__init__()

        self.lpips = pyiqa.create_metric("lpips", device=device, as_loss=True)
        self.mod = model.module.module if data_parallel else model.module
        self.all_combinations = self.mod.all_combinations

        self.wlpips = wlpips
        self.wadversary = wadversary
        self.beta = beta
        self.channel_weighting = channel_weighting

    def _m_dreg(self, model, x, K=1, test=False):
        qu_xs, px_us, uss = model("forward", x, K)

        loss_dict = {"lpips": 0, "recon_logprob": 0, "prior": 0}
        lw = 0
        for r, px_u in enumerate(px_us):
            wlpips_r = self.wlpips[r] if hasattr(self.wlpips, "__getitem__") else self.wlpips
            lpx_u = reconstruction_loss(px_u, x[r], self.mod.llik_scales[r], loss_dict, wlpips_r, self.lpips, self.channel_weighting[r])
            lw = lw + lpx_u

        for r, qu_x in enumerate(qu_xs):
            l_kl_div = gauss_kl_div_loss(qu_x.mean, qu_x.std).unsqueeze(0)
            lw = lw - self.beta * l_kl_div
            loss_dict["prior"] -= l_kl_div.detach().cpu().sum().item()

        return lw, uss, loss_dict

    def forward(self, model, x, K=1, test=False, **kwargs):
        S = compute_microbatch_split(x, K)
        x_split = zip(*[_x.split(S) for _x in x])

        lw, uss, loss_dicts = zip(*[self._m_dreg(model, _x, K, test, **kwargs) for _x in x_split])
        combined_loss_dict = {key: sum([loss_dict[key] for loss_dict in loss_dicts]) for key in loss_dicts[0].keys()}

        lw = torch.cat(lw, 2)  # concat on batch
        uss = torch.cat(uss, 2)  # concat on batch
        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
            if uss.requires_grad:
                uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return (grad_wt * lw).mean(0).sum(), combined_loss_dict