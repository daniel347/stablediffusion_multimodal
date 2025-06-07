import pytest
import torch
from torch import nn
import copy
from ldm.modules.ema import PowerFunctionEMA, TraditionalEMA, DummyEMA, solve_posthoc_coefficients
import numpy as np

ema_opts = {"power": {}, "traditional": {"halflife_Ming": 0.02}, "dummy": {}}


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(16))

    def forward(self, x):
        return x * self.param


def test_ema_update(ema_opts={}):
    model = TestModel()
    ema_opts["net"] = model
    ema = PowerFunctionEMA(**ema_opts)

    orig_p1_weights = copy.deepcopy(model.param.data)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in range(5):
        x = torch.randn(16)
        y = model(x)
        l = torch.linalg.vector_norm(y - x)
        l.backward()
        optim.step()

        ema.update(batch_size=1)

    assert torch.all(model.param.data != orig_p1_weights)

    for i in range(len(ema.emas)):
        assert torch.all(ema.emas[i]["param"].data != orig_p1_weights)
        assert torch.all(ema.emas[i]["param"].data != model.param.data)
        for j in range(i):
            assert torch.all(ema.emas[j]["param"].data != ema.emas[i]["param"].data)


def test_no_initial_params(ema_opts={}):
    model = TestModel()
    ema_opts["net"] = model
    ema = PowerFunctionEMA(**ema_opts)

    orig_p1_weights = copy.deepcopy(model.param.data)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(16)
    y = model(x)
    l = torch.linalg.vector_norm(y - x)
    l.backward()
    optim.step()
    ema.update(batch_size=1)

    assert torch.all(model.param.data != orig_p1_weights)

    for i in range(len(ema.emas)):
        assert torch.all(ema.emas[i]["param"].data != orig_p1_weights)
        # For 1 update we expect to not interpolate the initial state at all
        assert torch.all(ema.emas[i]["param"].data == model.param.data)
        for j in range(i):
            assert torch.all(ema.emas[j]["param"].data != ema.emas[i]["param"].data)


def test_power_ema_context(ema_opts={}):
    model = TestModel()
    ema_opts["net"] = model
    ema = PowerFunctionEMA(**ema_opts)

    rand_curr = torch.randn(16)
    model.param = nn.Parameter(rand_curr)

    rand_emas = [torch.randn(16) for _ in range(len(ema.emas))]
    for i in range(len(ema.emas)):
        ema.emas[i]["param"] = nn.Parameter(rand_emas[i])

    for i in range(len(ema.emas)):
        with ema.ema_scope(std_ind=i):
            assert torch.all(model.param.data == rand_emas[i])

        assert torch.all(model.param.data == rand_curr)


def test_solve_posthoc_coefficients():
    in_ofs = [10000, 20000, 30000]
    in_stds = [0.1, 0.1, 0.1]

    out_ofs = [20000]
    out_stds = [0.1]

    coeffs = solve_posthoc_coefficients(in_ofs, in_stds, out_ofs, out_stds)
    assert coeffs.shape == (3, 1)
    assert np.allclose(coeffs, np.array([[0], [1], [0]]))

