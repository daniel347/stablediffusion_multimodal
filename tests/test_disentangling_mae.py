import pytest
import torch
from mae.models_disentangling_mae import DisMAE
from objectives.disentangled_objective_mae import DisentanglingObjectiveMAE
from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
import matplotlib.pyplot as plt

vae_params = [{"target": "mae.models_mae.MaskedAutoencoderViT", "params": {}}, {"target": "mae.models_mae.MaskedAutoencoderViT", "params": {"in_chans": 18, "end_act": "sigmoid"}}]
model_params = {"vae_params": vae_params, "avg_z": False, "lik_dists": ["torch.distributions.Laplace", "torch.distributions.Bernoulli"], "lik_dist_params": [{"scale": 0.01}, {}]}
objective_params = {"wlpips": [1.0, 0.0], "wll": 1.0, "wadversary": 0.0, "start_adv_iter": 0, "beta": 1.0, "z_close_weight": 50000, "mmd_zw_weight": 1, "mmd_ww_weight": 1, "channel_weighting": [None, None]}

@pytest.mark.parametrize("mask_fraction, mask_overlap", ([0.25, 0.5], [0.5, 0.8], [0.75, 0.9]))
def test_masking(mask_fraction, mask_overlap):
    model = DisMAE(**model_params)
    x_shape = (1, 196, 1024)
    x_device = torch.device("cpu")
    masks, ids_restore, ids_keep_unique, ids_keep_shared, len_mask_unique = model.overlapped_masking(x_shape, x_device, mask_fraction, mask_overlap)

    print(f"mask sums{[m.sum() for m in masks]}")
    print(f"overlap: {(masks[0] * masks[1]).sum()}")

    masks_reshaped = [mask.reshape(14, 14) for mask in masks]
    masks_reshaped.append(torch.zeros(14, 14))
    masks_reshaped = torch.stack(masks_reshaped, dim=-1)


    plt.imshow(masks_reshaped)
    plt.savefig(f"masks_reshaped_{mask_fraction}_{mask_overlap}.png")

    print(masks)
    print(ids_restore)


def test_model_encode():
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = [torch.randn(1, 3, 224, 224).to(x_device), torch.clamp(torch.randn(1, 18, 224, 224), 0, 1).to(x_device)]
    model = DisMAE(**model_params).to(x_device)

    W, Z, ids_restore, cls_tokens, len_keep_shared, len_mask_unique = model.encode(data, total_mask_ratio=0.25, shared_mask_ratio=0.5)

    print(f"W shapes: {[w.shape for w in W]}")
    print(f"Z shapes: {[z.shape for z in Z]}")
    print(f"ids_restore shape: {ids_restore.shape}")
    print(f"cls_token: {[ct.shape for ct in cls_tokens]}")
    print(f"len_keep_shared: {len_keep_shared}, len_mask_unique: {len_mask_unique}")

    for w in W:
        assert w.shape[-1] == model.w_len
        assert w.shape[-2] == len_keep_shared - len_mask_unique * (model.n_modality - 1)

    for z in Z:
        assert z.shape[-1] == model.z_len
        assert z.shape[-2] == len_keep_shared - len_mask_unique * (model.n_modality - 1)

    for ct in cls_tokens:
        assert ct.shape == torch.Size([1, 1, model.u_len])

    assert ids_restore.shape[-1] == 196

@pytest.mark.parametrize("mask_fraction, mask_overlap, random_z", ([0.25, 0.5, False], [0.5, 0.8, False], [0.75, 0.9, False], [0.25, 0.5, True], [0.5, 0.8, True], [0.75, 0.9, True]))
def test_model_z_comb(mask_fraction, mask_overlap, random_z):
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = [torch.randn(1, 3, 224, 224).to(x_device), torch.clamp(torch.randn(1, 18, 224, 224), 0, 1).to(x_device)]
    model = DisMAE(**model_params).to(x_device)

    W, Z, ids_restore, cls_tokens, len_keep_shared, len_mask_unique = model.encode(data, total_mask_ratio=mask_fraction, shared_mask_ratio=mask_overlap)
    print(ids_restore)
    z_comb = model.z_combine(Z, len_keep_shared, len_mask_unique, ids_restore, avg_z=(not random_z), random_z=random_z)

    print(f"Z shapes: {[z.shape for z in Z]}")
    print(f"z_comb shape: {z_comb.shape}")

    for z in Z:
        assert z.shape[-1] == model.z_len
    assert z_comb.shape[-1] == model.z_len

    assert z_comb.shape[1] == len_keep_shared


def test_model_forward():
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = [torch.randn(1, 3, 224, 224).to(x_device), torch.clamp(torch.randn(1, 18, 224, 224), 0, 1).to(x_device)]
    model = DisMAE(**model_params).to(x_device)

    W, Z, z_comb, X = model.forward(data, random_z=True)

    print(f"W shapes: {[w.shape for w in W]}")
    print(f"Z shapes: {[z.shape for z in Z]}")
    print(f"z_comb shape: {z_comb.shape}")
    print(f"X shapes: {[x.shape for x in X]}")

    assert all([x.shape == data[i].shape for i, x in enumerate(X)])


def test_objective():
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = [torch.randn(1, 3, 224, 224).to(x_device), (torch.randn(1, 18, 224, 224) > 0).to(dtype=torch.float32, device=x_device)]
    model = MMVAEDataParallelWrapper(DisMAE(**model_params).to(x_device))
    objective = DisentanglingObjectiveMAE(model, x_device, **objective_params)

    wlw, ld, mo = objective(model, data, model_params=dict(total_mask_ratio=0.5, shared_mask_ratio=0.5, random_z=True))

    print(f"wlw shape: {wlw.shape}")
    print(f"loss dict: {ld}")