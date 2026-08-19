import pytest
from SiT.sit_model import SiTModel
import yaml
import torch
from omegaconf import OmegaConf

model_params = OmegaConf.load("configs/test-sit-mm-rgb.yaml").params

def construct_sit_model():
    model = SiTModel(**model_params)

@pytest.mark.parametrize("patch_size, in_channels, n_cls", ([1, 768, 0], [2, 768, 0], [2, 6, 0],
                                                     [1, 768, 1], [2, 768, 1],
                                                     [1, 768, 2], [2, 768, 2],))
def test_forward(patch_size, in_channels, n_cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_params = model_params.copy()
    test_params["scorenet_config"]["params"]["patch_size"] = patch_size
    test_params["scorenet_config"]["params"]["in_channels"] = in_channels
    test_params["scorenet_config"]["params"]["n_cls_tokens"] = n_cls

    model = SiTModel(**test_params)
    model.to(device)

    x = torch.randn(1, in_channels, 16, 16).to(device)
    c = torch.randn(1, n_cls, model.model.hidden_size).to(device) if n_cls > 0 else None

    loss, loss_dict = model(x, cls=c)

    assert isinstance(loss, torch.Tensor)
    assert "loss" in loss_dict


@pytest.mark.parametrize("patch_size, in_channels, n_cls", ([1, 768, 0], [2, 768, 0], [2, 6, 0],
                                                     [1, 768, 1], [2, 768, 1],
                                                     [1, 768, 2], [2, 768, 2],))
def test_q_sample(patch_size, in_channels, n_cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_params = model_params.copy()
    test_params["scorenet_config"]["params"]["patch_size"] = patch_size
    test_params["scorenet_config"]["params"]["in_channels"] = in_channels
    test_params["scorenet_config"]["params"]["n_cls_tokens"] = n_cls

    model = SiTModel(**test_params)
    model.to(device)

    x = torch.randn(1, in_channels, 16, 16).to(device)
    c = torch.randn(1, n_cls, model.model.hidden_size).to(device) if n_cls > 0 else None
    t = torch.randint(0, 1000, (1,)).to(device)

    out = model.q_sample(x, cls=c, t=t)
    if n_cls > 0:
        xt, cls_t = out
        assert cls_t.shape == c.shape
        assert xt.shape == x.shape

    else:
        xt = out
        assert xt.shape == x.shape


@pytest.mark.parametrize("patch_size, in_channels, n_cls", ([1, 768, 0], [2, 768, 0], [2, 6, 0],
                                                     [1, 768, 1], [2, 768, 1],
                                                     [1, 768, 2], [2, 768, 2],))
def test_sample(patch_size, in_channels, n_cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_params = model_params.copy()
    test_params["scorenet_config"]["params"]["patch_size"] = patch_size
    test_params["scorenet_config"]["params"]["in_channels"] = in_channels
    test_params["scorenet_config"]["params"]["n_cls_tokens"] = n_cls

    model = SiTModel(**test_params)
    model.to(device)
    b = 1

    samples = model.sample(verbose=False, shape=[in_channels, 16, 16], batch_size=b, n_cls=n_cls)
    print(f"samples type: {type(samples)}")
    if n_cls > 0:
        print(f"sample ps len: {len(samples)}, shape: {[s.shape for s in samples]}")
        x_samples = samples[0]
        cls_samples = samples[1]
        assert cls_samples.shape == (b, n_cls, model.model.hidden_size)
        assert x_samples.shape == torch.Size((b, in_channels, 16, 16))
    else:
        assert samples.shape == torch.Size((b, in_channels, 16, 16))





