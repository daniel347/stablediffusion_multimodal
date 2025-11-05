import pytest
from SiT.sit_model import SiTModel
import yaml
import torch

with open("configs/model/mae-sit-mm-rgb.yaml", 'r') as f:
    model_params = yaml.safe_load(f)["params"]

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

    samples = model.sample(verbose=False, shape=[in_channels, 16, 16], batch_size=2, n_cls=n_cls)
    print(f"samples len: {len(samples)}, shape: {[s.shape for s in samples]}")
    if n_cls > 0:
        x_samples = samples[0]
        cls_samples = samples[1]
        assert cls_samples.shape == (2, 1, model.embed_dim)
        assert x_samples.shape == torch.Size((2, in_channels, 16, 16))





