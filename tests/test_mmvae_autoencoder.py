import pytest
from ldm.util import instantiate_from_config, dict_to_namespace
import torch
import yaml
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torch.distributions import Laplace


def construct_mmvae():
    with open("tests/configs/test_mmvae_autoencoder.yaml") as f:
        config = yaml.safe_load(f)

    return instantiate_from_config(config), dict_to_namespace(config)


def test_to_device():
    mmvae, conf = construct_mmvae()

    if torch.cuda.is_available():
        mmvae = mmvae.to("cuda")
        mmvae = mmvae.to("cpu")


@pytest.mark.parametrize("K, forward_f", [(1, "forward"),
                                          (3, "forward"),
                                          (1, "self_and_cross_modal_generation_forward"),
                                          (3, "self_and_cross_modal_generation_forward")])
def test_forward(K, forward_f):
    """This tests both the forward function and the self_and_cross_modal_generation_forward function,
     which uses different prior distributions"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mmvae, conf = construct_mmvae()
    mmvae = mmvae.to(device)

    zw_mean_shape = (K,
                     1,
                     conf.params.latent_channel_z + conf.params.latent_channel_w,
                     conf.params.vae_params[0].params.embed_spatial_dim,
                     conf.params.vae_params[0].params.embed_spatial_dim)

    X = [torch.randn(1, 3, 256, 256).to(device), torch.randn(1, 1, 256, 256).to(device)]

    with torch.no_grad():
        qu_xs, px_us, uss = getattr(mmvae, forward_f)(X, K=K)

    assert len(qu_xs) == 2
    for qu_x in qu_xs:
        assert isinstance(qu_x, DiagonalGaussianDistribution)
        assert qu_x.mean.shape == zw_mean_shape[1:]

    assert len(px_us) == 2
    for px_u in px_us:
        assert len(px_u) == 2
        for m, px in enumerate(px_u):
            assert isinstance(px, Laplace)
            assert px.mean.shape[1:] == X[m].shape
            assert px.mean.shape[0] == K

    assert len(uss) == 2
    for us in uss:
        assert us.shape == zw_mean_shape


@pytest.mark.parametrize("N", [1, 5])
def test_generate_unconditional(N):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mmvae, conf = construct_mmvae()
    mmvae = mmvae.to(device)

    with torch.no_grad():
        gens = mmvae.generate_unconditional(N)

    assert len(gens) == 2
    assert gens[0].shape == (N, 3, 256, 256)
    assert gens[1].shape == (N, 1, 256, 256)


def test_self_and_cross_modal_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mmvae, conf = construct_mmvae()
    mmvae = mmvae.to(device)

    X = [torch.randn(1, 3, 256, 256).to(device), torch.randn(1, 1, 256, 256).to(device)]

    with torch.no_grad():
        gens = mmvae.self_and_cross_modal_generation(X)

    assert len(gens) == 2
    for gen in gens:
        assert len(gens) == 2
        for m, g in enumerate(gen):
            assert g.shape == (1, *X[m].shape)


def test_latent_parameters():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mmvae, conf = construct_mmvae()
    mmvae = mmvae.to(device)

    pz_params = mmvae.pz_params
    pw_params = mmvae.pw_params

    z_mean_shape = (1,
                     conf.params.latent_channel_z,
                     conf.params.vae_params[0].params.embed_spatial_dim,
                     conf.params.vae_params[0].params.embed_spatial_dim)

    w_mean_shape = (1,
                     conf.params.latent_channel_w,
                     conf.params.vae_params[0].params.embed_spatial_dim,
                     conf.params.vae_params[0].params.embed_spatial_dim)

    assert pz_params[0].shape == z_mean_shape
    assert pz_params[1].shape == z_mean_shape

    assert pw_params[0].shape == w_mean_shape
    assert pw_params[1].shape == w_mean_shape

