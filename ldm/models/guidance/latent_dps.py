import torch.linalg
from torch import nn

class LatentDPS(nn.Module):

    def __init__(self, model_first_stage, measurement_models, mod_weights, scale):
        super().__init__()
        self.model_first_stage = model_first_stage
        self.meas_models = measurement_models
        self.mod_weights = mod_weights
        self.scale = scale

    def forward(self, zprev, zt, z0, y, t, **kwargs):
        z0_scaled = self.model_first_stage.scale_latents(z0, reverse=True)
        x0 = self.model_first_stage.decode(z0_scaled)

        meas_error = 0
        for i, (y_mod, x0_mod) in enumerate(zip(y, x0)):
            if y_mod is None:
                continue

            x0_meas = self.meas_models[i](x0_mod.mean)
            meas_error = meas_error + self.mod_weights[i] * torch.linalg.vector_norm(y_mod - x0_meas, dim=(1, 2, 3))

        norm_grads = torch.autograd.grad(meas_error.sum(), zt)[0]
        return zprev - self.scale * norm_grads


class IdentityMeasurement(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x