import pytest
from ldm.modules.diffusionmodules.model import Encoder, Decoder
import torch
import torch.utils.checkpoint
import yaml
from functools import partial
import copy


def construct_model(model):
    with open("tests/configs/test_encoder_decoder.yaml") as f:
        config = yaml.safe_load(f)

    return model(**config), config



def test_encoder_grad_checkpointing():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc, conf = construct_model(Encoder)
    enc = enc.to(device)

    x = torch.randn(1, 3, 256, 256).to(device)
    x.requires_grad = True

    out = enc(x)

    assert out.shape == (1, 12, 32, 32)

    # Just compute some gradients
    out.mean().backward()

    # This could be a bit strict
    for p in enc.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Now try with checkpointing
    enc2 = copy.deepcopy(enc)
    enc2 = enc2.to(device)
    enc2.checkpoint_fn = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    out2 = enc2(x)
    out2.mean().backward()

    assert torch.allclose(out, out2)

    for p1, p2 in zip(enc.parameters(), enc2.parameters()):
        if p2.requires_grad:
            assert p2.grad is not None
            assert torch.allclose(p1.grad, p2.grad)


def test_decoder_grad_checkpointing():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dec, conf = construct_model(Decoder)
    dec = dec.to(device)

    z = torch.randn(1, 6, 32, 32).to(device)
    z.requires_grad = True

    out = dec(z)

    assert out.shape == (1, 3, 256, 256)

    # Just compute some gradients
    out.mean().backward()

    # This could be a bit strict
    for p in dec.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # Now try with checkpointing
    dec2 = copy.deepcopy(dec)
    dec2 = dec2.to(device)
    dec2.checkpoint_fn = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

    out2 = dec2(z)
    out2.mean().backward()

    assert torch.allclose(out, out2)

    for p1, p2 in zip(dec.parameters(), dec2.parameters()):
        if p2.requires_grad:
            assert p2.grad is not None
            assert torch.allclose(p1.grad, p2.grad)









