import pytest
from mae.distribution import DiagonalGaussian
from torch.distributions import Normal
import torch
import numpy as np


def test_dist():
    a = torch.randn(3, 64, 768)
    d = DiagonalGaussian(a)

    assert d.batch_shape == torch.Size([3])
    assert d.event_shape == torch.Size([64, 384])
    assert d.shape == torch.Size([3, 64, 384])

    assert d.mean.shape == torch.Size([3, 64, 384])
    assert d.logvar.shape == torch.Size([3, 64, 384])

def test_sample():
    a = torch.randn(3, 64, 768)
    d = DiagonalGaussian(a)

    s1 = d.sample()
    s2 = d.rsample()

    assert s1.shape == torch.Size([3, 64, 384])
    assert s2.shape == torch.Size([3, 64, 384])

def test_log_prob():
    a = torch.randn(3, 64, 768)
    d = DiagonalGaussian(a)

    s = d.sample()
    lp = d.log_prob(s)

    assert lp.shape == torch.Size([3, 64, 384])

def test_add():
    a = torch.randn(3, 64, 768)
    b = torch.randn(3, 64, 768)

    d1 = DiagonalGaussian(a)
    d2 = DiagonalGaussian(b)

    d3 = d1 + d2

    assert d3.mean.shape == torch.Size([3, 64, 384])
    assert d3.logvar.shape == torch.Size([3, 64, 384])

    assert torch.allclose(d3.mean, d1.mean + d2.mean)
    assert torch.allclose(d3.variance, d1.variance + d2.variance)

def test_sub():
    a = torch.randn(3, 64, 768)
    b = torch.randn(3, 64, 768)

    d1 = DiagonalGaussian(a)
    d2 = DiagonalGaussian(b)

    d3 = d1 - d2

    assert d3.mean.shape == torch.Size([3, 64, 384])
    assert d3.logvar.shape == torch.Size([3, 64, 384])

    assert torch.allclose(d3.mean, d1.mean - d2.mean)
    assert torch.allclose(d3.variance, d1.variance + d2.variance)

def test_slice():
    a = torch.randn(3, 64, 768)
    d = DiagonalGaussian(a)

    d2 = d[1]

    assert d2.mean.shape == torch.Size([64, 384])
    assert d2.logvar.shape == torch.Size([64, 384])

    d3 = d[:, 32:48, :]

    assert d3.mean.shape == torch.Size([3, 16, 384])
    assert d3.logvar.shape == torch.Size([3, 16, 384])
    assert torch.allclose(d3.mean, d.mean[:, 32:48, :])

def test_split_cat():
    a = torch.randn(3, 64, 768)
    d = DiagonalGaussian(a)

    d1, d2 = torch.split(d, [256, 128], dim=-1)
    assert d1.mean.shape == torch.Size([3, 64, 256])
    assert d2.mean.shape == torch.Size([3, 64, 128])

    d_back = DiagonalGaussian.cat_dists((d1, d2), dim=-1)
    assert d_back.mean.shape == torch.Size([3, 64, 384])
    assert torch.allclose(d_back.mean, d.mean)

    assert torch.allclose(d_back.variance, d.variance)

def test_mul_div():
    a = torch.randn(3, 64, 768)

    d1 = DiagonalGaussian(a)

    d2 = d1 / 3.0
    d3 = d1 * 2.5

    assert d2.mean.shape == torch.Size([3, 64, 384])
    assert d2.logvar.shape == torch.Size([3, 64, 384])

    assert d3.mean.shape == torch.Size([3, 64, 384])
    assert d3.logvar.shape == torch.Size([3, 64, 384])

    assert torch.allclose(d2.mean, d1.mean / 3.0)
    assert torch.allclose(d2.variance, d1.variance / 9.0)
    assert torch.allclose(d3.mean, d1.mean * 2.5)
    assert torch.allclose(d3.variance, d1.variance * 6.25)


def test_assign_slice():
    a = torch.randn(3, 64, 768)
    b = torch.randn(3, 32, 768)
    inds = torch.randperm(64)[:32]

    d1 = DiagonalGaussian(a)
    d2 = DiagonalGaussian(b)

    d1.assign_slice(np.s_[:, inds, :], d2)

    assert torch.allclose(d1.mean[:, inds, :], d2.mean)
    assert torch.allclose(d1.variance[:, inds, :], d2.variance)


def test_kl_divergence():
    a = torch.randn(3, 64, 768)
    b = torch.randn(3, 64, 768)

    d1 = DiagonalGaussian(a)
    d2 = DiagonalGaussian(b)

    kl = torch.distributions.kl.kl_divergence(d1, d2)

    true_kl = torch.distributions.kl.kl_divergence(Normal(a[:, :, :384], torch.exp(a[:, :, 384:]/2.0)),
                                                   Normal(b[:, :, :384], torch.exp(b[:, :, 384:]/2.0))).sum(dim=(-1, -2))

    assert torch.allclose(kl, true_kl)
    assert kl.shape == torch.Size([3])