import pytest
import torch
from mae.models_disentangling_mae import DisMAE
import matplotlib.pyplot as plt

vae_params = [{"target": "mae.models_mae.MaskedAutoencoderViT", "params": {}} for _ in range(2)]

@pytest.mark.parametrize("mask_fraction, mask_overlap", ([0.25, 0.5], [0.5, 0.8], [0.75, 0.9]))
def test_masking(mask_fraction, mask_overlap):
    model = DisMAE(vae_params, avg_z=False)
    x_shape = (1, 196, 1024)
    x_device = torch.device("cpu")
    masks, ids_restore = model.overlapped_masking(x_shape, x_device, mask_fraction, mask_overlap)

    print(f"mask sums{[m.sum() for m in masks]}")
    print(f"overlap: {(masks[0] * masks[1]).sum()}")

    masks_reshaped = [mask.reshape(14, 14) for mask in masks]
    masks_reshaped.append(torch.zeros(14, 14))
    masks_reshaped = torch.stack(masks_reshaped, dim=-1)


    plt.imshow(masks_reshaped)
    plt.savefig(f"masks_reshaped_{mask_fraction}_{mask_overlap}.png")

    print(masks)
    print(ids_restore)


