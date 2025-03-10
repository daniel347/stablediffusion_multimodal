import pytest
from ldm.util import instantiate_from_config, dict_to_namespace
import torch
import yaml
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torch.distributions import Laplace


def construct_vae(config_path="tests/configs/test_autoencoder.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return instantiate_from_config(config), dict_to_namespace(config)


def test_to_device():
    vae, conf = construct_vae()

    if torch.cuda.is_available():
        vae = vae.to("cuda")
        vae = vae.to("cpu")


def test_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, conf = construct_vae()
    vae.to(device)

    x = torch.randn(1, 3, 256, 256).to(device)

    zw_dist = vae.encode(x)

    zw_mean_shape = (1,
                     conf.params.ddconfig.z_channels,
                     conf.params.embed_spatial_dim,
                     conf.params.embed_spatial_dim)


    assert isinstance(zw_dist, DiagonalGaussianDistribution)
    assert zw_dist.mean.shape == zw_mean_shape
    assert zw_dist.logvar.shape == zw_mean_shape


def test_decoder_K1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, conf = construct_vae()
    vae.to(device)

    x = torch.randn(1, 3, 256, 256).to(device)

    zw_dist = vae.encode(x)
    x_recon = vae.decode(zw_dist.sample())

    assert isinstance(x_recon, Laplace)
    assert x_recon.mean.shape == x.shape


def test_decoder_K5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, conf = construct_vae()
    vae.to(device)

    x = torch.randn(1, 3, 256, 256).to(device)

    zw_dist = vae.encode(x)
    zw_samp = zw_dist.sample(5)
    x_recon = vae.decode(zw_samp)

    assert isinstance(x_recon, Laplace)

    x_recon = x_recon.mean

    assert x_recon.shape[1:] == x.shape
    assert x_recon.shape[0] == 5


def test_init_with_sd():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, conf = construct_vae("tests/configs/test_autoencoder_conform_to_3_channel.yaml")

    sd_state_dict = torch.load("/home/ds3119/multimodal_generation/stablediffusion/checkpoints/v2-1_512-ema-pruned.ckpt")
    first_stage_dict = {}

    for name, param in sd_state_dict["state_dict"].items():
        if "first_stage_model" in name:
            new_name = name.split("first_stage_model.")[1]
            first_stage_dict[new_name] = param
    M, U = vae.load_state_dict(first_stage_dict, strict=False)
    for m in M:
        print(f"Missing key: {m}")
    for u in U:
        print(f"Unused key: {u}")

    assert len(U) == 0
    assert len(M) == 2  # Just the auxiliary w distribution parameters

    vae.to(device)

    x = torch.randn(1, 1, 256, 256).to(device)

    zw_dist = vae.encode(x)
    zw_samp = zw_dist.sample(5)
    x_recon = vae.decode(zw_samp)

    assert isinstance(x_recon, Laplace)

    x_recon = x_recon.mean

    assert x_recon.shape[1:] == x.shape
    assert x_recon.shape[0] == 5


@pytest.mark.parametrize("K", [1, 3])
def test_forward(K):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, conf = construct_vae()
    vae.to(device)
    x = torch.randn(1, 3, 256, 256).to(device)

    post, dec, z = vae.forward(x, K=K)

    zw_mean_shape = (K,
                     1,
                     conf.params.ddconfig.z_channels,
                     conf.params.embed_spatial_dim,
                     conf.params.embed_spatial_dim)

    assert isinstance(post, DiagonalGaussianDistribution)
    assert isinstance(dec, Laplace)
    assert z.shape == zw_mean_shape

    assert vae.qu_x_params is not None
    assert vae.qu_x_params.mean.shape == zw_mean_shape[1:]

