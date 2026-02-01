import torch.linalg
from torch import nn

class LatentDPS(nn.Module):

    def __init__(self, model_first_stage, measurement_models, mod_weights, scale, manifold_constraint=False, curvature_bound=None):
        super().__init__()
        self.model_first_stage = model_first_stage
        self.meas_models = measurement_models
        self.mod_weights = mod_weights
        self.scale = scale
        self.manifold_constraint = manifold_constraint
        self.curvature_bound = curvature_bound

    def forward(self, zprev, zt, z0, y, t, et, return_correction=False, **kwargs):
        z0_scaled = self.model_first_stage.scale_latents(z0, reverse=True)
        x0 = self.model_first_stage.decode(z0_scaled)

        meas_error = 0
        for i, (y_mod, x0_mod) in enumerate(zip(y, x0)):
            if y_mod is None:
                continue

            x0_meas = self.meas_models[i](x0_mod.mean)
            meas_error = meas_error + self.mod_weights[i] * torch.linalg.vector_norm(y_mod - x0_meas, dim=(1, 2, 3))

        norm_grads = torch.autograd.grad(meas_error.sum(), zt, create_graph=(self.curvature_bound is not None))[0]

        scale = self.scale
        if self.manifold_constraint:
            n_hat = et/torch.linalg.norm(et, dim=(1, 2, 3), keepdim=True)
            tangent_manifold_error = torch.sum(zt.grad * n_hat, dim=(1, 2, 3), keepdim=True)
            print(f"tangent_manifold_error: {tangent_manifold_error}")
            norm_grads = norm_grads - tangent_manifold_error * norm_grads

            if self.curvature_bound is not None:
                n_hat.backward(gradient=norm_grads)
                curvature_manifold_error = torch.sum(zt.grad * n_hat, dim=(1, 2, 3), keepdim=True)
                print(f"curvature_manifold_error: {curvature_manifold_error}")
                scale = torch.clamp((self.curvature_bound / curvature_manifold_error).sqrt(), min=None, max=self.scale)

        if return_correction:
            return -scale * norm_grads
        return zprev - scale * norm_grads