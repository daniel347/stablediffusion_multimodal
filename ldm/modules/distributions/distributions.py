import torch
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, K=1, squeeze=True):
        sample_shape = [K, *self.mean.shape] if K > 1 or squeeze == False else self.mean.shape
        x = self.mean + self.std * torch.randn(sample_shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

    def detach(self):
        """ Detatches mean, variance etc from computational graph"""
        self.mean, self.logvar = self.mean.detach(), self.logvar.detach()
        self.std = self.std.detach()
        self.var = self.var.detach()

    def scatter(self, target_gpus, dim=0):
        if len(self.mean.shape) == 5:
            dim += 1  # We dont want to split on the K dimension if it exists
        mean_scattered = torch.nn.parallel.scatter(self.mean, target_gpus, dim)
        logvar_scattered = torch.nn.parallel.scatter(self.logvar, target_gpus, dim)
        return [DiagonalGaussianDistribution(m, l) for m, l in zip(mean_scattered, logvar_scattered)]

    def gather(self, target_device, dim=0):
        if len(self.mean.shape) == 5:
            dim += 1  # We dont want to merge on the K dimension if it exists
        mean_gathered = torch.nn.parallel.gather(self.mean, target_device, dim)
        logvar_gathered = torch.nn.parallel.gather(self.logvar, target_device, dim)
        return DiagonalGaussianDistribution(mean_gathered, logvar_gathered)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
