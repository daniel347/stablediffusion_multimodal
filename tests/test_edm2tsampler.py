import pytest
import torch
from ldm.util import instantiate_from_config, dict_to_namespace
from ldm.modules.diffusionmodules.util import make_beta_schedule
import yaml
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def constuct_alphas_cumprod(beta_schedule="linear",
                      timesteps=1000,
                      linear_start=1e-4,
                      linear_end=2e-2,
                      cosine_s=8e-3):

    betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s = cosine_s)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    return torch.from_numpy(alphas_cumprod)


def construct_tsampler(config_path="tests/configs/test_edm2tsampler.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        alphas_cumprod = constuct_alphas_cumprod()
        config["params"]["sigma_sched"] = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)

    return instantiate_from_config(config), dict_to_namespace(config)


def test_sample_t():
    tsampler, config = construct_tsampler()
    imgs = torch.randn(32, 6, 256, 256)

    t = tsampler(imgs)

    assert t.shape[0] == 32


def test_sample_t_dist():
    tsampler, config = construct_tsampler()
    imgs = torch.randn(32, 6, 256, 256)

    ts = []
    for _ in range(100):
        ts.append(tsampler(imgs))

    ts = torch.stack(ts)

    hist = sns.histplot(x=ts.flatten().cpu(), bins=25, kde=True)
    fig = hist.get_figure()
    fig.savefig("t_histogram.png")
    plt.clf()


def test_edm_loss_weighting():
    alphas_cumprod = constuct_alphas_cumprod()
    sigma_data = 0.5
    # This code is copied from the DDPM class because im too lazy to initialise the whole class
    sigma = torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

    ts = torch.arange(100, 1000, 1)

    plot = sns.lineplot(x=ts, y=weight[100:], )
    plot.set(xlabel='diffusion timestep', ylabel='loss weighting')
    fig = plot.get_figure()
    fig.savefig("weighting.png")
    plt.clf()
