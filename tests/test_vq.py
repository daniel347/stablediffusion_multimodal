import pytest
from mae.vector_quantiser import VectorQuantiser
import torch

quantiser_opts = {"n_e": 8196, "e_dim": 32, "beta": 1, "e_dim_in": 768, "normalise": False, "resample_keep_thresh": 5e-3, "resample_noise": 0.1, "temp": 0.01}
B = 2
L = 64

@pytest.mark.parametrize("warm_start", (True, False))
def test_quantise(warm_start):
    vq = VectorQuantiser(**quantiser_opts)

    z = torch.randn(B, L, 768, requires_grad=True)

    loss, z_q, perplexity, min_encodings, min_encoding_indices = vq.forward(z, warm_start)

    assert z.shape == z_q.shape
    assert loss.item()
    assert min_encodings.shape == (*z.shape[:-1], quantiser_opts["n_e"])
    assert min_encoding_indices.shape == (*z.shape[:-1], 1)

    if warm_start:
        assert torch.allclose(z_q, z)  # no quantisation performed
        loss.backward()
        print(z.grad)
        assert z.grad is None
    else:
        loss.backward()
        print(z.grad)
        assert z.grad is not None
