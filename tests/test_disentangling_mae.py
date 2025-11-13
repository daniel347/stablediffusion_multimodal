import pytest
import torch
from mae.models_disentangling_mae import DisMAE
from mae.distribution import DiagonalGaussian
from objectives.disentangled_objective_mae import DisentanglingObjectiveMAE
from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
import matplotlib.pyplot as plt
import math
import yaml

with open("../configs/model/mae-sit-mm-rgb-semseg.yaml", 'r') as f:
    model_params = yaml.safe_load(f)["params"]["first_stage_config"]["params"]

with open('../configs/model/mae-sit-mm-rgb.yaml', 'r') as f:
    unimodal_model_params = yaml.safe_load(f)["params"]["first_stage_config"]["params"]

objective_params = {"wlpips": [1.0, 0.0], "wll": 1.0, "wadversary": 0.0, "start_adv_iter": 0, "beta": 1.0, "z_close_weight": 50000, "mmd_zw_weight": 1, "mmd_ww_weight": 1, "channel_weighting": [None, None]}
unimodal_objective_params = {"wlpips": [1.0], "wll": 1.0, "wadversary": 0.0, "start_adv_iter": 0, "beta": 1.0, "z_close_weight": 0, "mmd_zw_weight": 1, "mmd_ww_weight": 0, "channel_weighting": [None]}

img_size = 128

@pytest.mark.parametrize("mask_fraction, mask_overlap, unimodal", ([0.25, 0.5, False], [0.5, 0.8, False], [0.75, 0.9, False], [0.25, 0.5, True], [0.5, 0.8, True], [0.75, 0.9, True]))
def test_masking(mask_fraction, mask_overlap, unimodal):
    model = DisMAE(**(unimodal_model_params if unimodal else model_params))

    n_patches = model.vaes[0].patch_embed.num_patches
    mask_dim = int(math.sqrt(n_patches))

    x_shape = (1, n_patches, 1024)
    x_device = torch.device("cpu")
    masks, ids_restore, ids_keep_unique, ids_keep_shared, len_mask_unique = model.overlapped_masking(x_shape, x_device, mask_fraction, mask_overlap)

    print(f"mask sums{[m.sum() for m in masks]}")
    for mask in masks:
        assert mask.sum() == int(n_patches * mask_fraction), f"Mask sum {mask.sum()} does not match expected {int(n_patches * (1 - mask_fraction))}"
    if len(masks) > 1:
        overlaps = [[0 for _ in range(len(masks))] for _ in range(len(masks))]
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                overlaps[i][j] = (masks[i] * masks[j]).sum()
                assert abs(overlaps[i][j] - int(n_patches * mask_fraction * mask_overlap)) <= 1, f"Mask overlap {overlaps[i][j]} does not match expected value {int(n_patches * mask_fraction * mask_overlap)}"
        print(f"overlaps: {overlaps}")

    # Check if ids_restore and mask are compatible
    for i, mask in enumerate(masks):
        assert torch.all(ids_restore[mask.to(bool)] >= n_patches * (1 - mask_fraction) + len_mask_unique * (i - 1)), f"ids_restore and mask {i} are not compatible"

    masks_reshaped = [mask.reshape(mask_dim, mask_dim) for mask in masks]

    while len(masks_reshaped) < 3:
        # force to have 3 channels for visualization as RGB
        masks_reshaped.append(torch.zeros(mask_dim, mask_dim))
    masks_reshaped = torch.stack(masks_reshaped, dim=-1)

    plt.imshow(masks_reshaped)
    plt.savefig(f"masks_reshaped_{mask_fraction}_{mask_overlap}_{'unimodal' if unimodal else ''}.png")

@pytest.mark.parametrize("unimodal, is_variational", ([False, False], [True, False], [False, True], [True, True]))
def test_model_encode(unimodal, is_variational):
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if unimodal:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device)]
        model = DisMAE(is_variational=is_variational, **unimodal_model_params).to(x_device)
    else:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device), torch.clamp(torch.randn(1, 18, img_size, img_size), 0, 1).to(x_device)]
        model = DisMAE(is_variational=is_variational, **model_params).to(x_device)

    W, Z, ids_restore, cls_tokens, len_keep_shared, len_mask_unique, masks = model.encode(data, total_mask_ratio=0.25, shared_mask_ratio=0.5)

    print(f"W shapes: {[w.shape for w in W]}")
    print(f"Z shapes: {[z.shape for z in Z]}")
    print(f"ids_restore shape: {ids_restore.shape}")
    print(f"cls_token: {[ct.shape for ct in cls_tokens]}")
    print(f"len_keep_shared: {len_keep_shared}, len_mask_unique: {len_mask_unique}")

    for w in W:
        assert w.shape[-1] == model.w_len
        assert w.shape[-2] == len_keep_shared - len_mask_unique

    for z in Z:
        assert z.shape[-1] == model.z_len
        assert z.shape[-2] == len_keep_shared - len_mask_unique

    for ct in cls_tokens:
        assert ct.shape == torch.Size([1, 1, model.u_len])

    if is_variational:
        assert all([isinstance(z, DiagonalGaussian) for z in Z])
        assert all([isinstance(w, DiagonalGaussian) for w in W])
        assert all([isinstance(ct, DiagonalGaussian) for ct in cls_tokens])

    assert ids_restore.shape[-1] == model.vaes[0].patch_embed.num_patches

@pytest.mark.parametrize("mask_fraction, mask_overlap, random_z, unimodal, is_variational",
                         ([0.25, 0.5, False, True, False], [0.5, 0.8, False, True, False], [0.75, 0.9, False, True, True], [0.25, 0.5, True, True, True], [0.5, 0.8, True, True, False], [0.75, 0.9, True, True, False],
                          [0.25, 0.5, False, False, False], [0.5, 0.8, False, False, True], [0.75, 0.9, False, False, False], [0.25, 0.5, True, False, False], [0.5, 0.8, True, False, True], [0.75, 0.9, True, False, True]))
def test_model_z_comb(mask_fraction, mask_overlap, random_z, unimodal, is_variational):
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if unimodal:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device)]
        model = DisMAE(is_variational=is_variational, **unimodal_model_params).to(x_device)
    else:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device), torch.clamp(torch.randn(1, 18, img_size, img_size), 0, 1).to(x_device)]
        model = DisMAE(is_variational=is_variational, **model_params).to(x_device)

    W, Z, ids_restore, cls_tokens, len_keep_shared, len_mask_unique, masks = model.encode(data, total_mask_ratio=mask_fraction, shared_mask_ratio=mask_overlap)
    print(ids_restore)
    z_comb = model.z_combine(Z, len_keep_shared, len_mask_unique, ids_restore, avg_z=(not random_z), random_z=random_z)

    print(f"Z shapes: {[z.shape for z in Z]}")
    print(f"z_comb shape: {z_comb.shape}")

    for z in Z:
        assert z.shape[-1] == model.z_len
    assert z_comb.shape[-1] == model.z_len

    if unimodal:
        if is_variational:
            assert torch.allclose(z_comb.mean, Z[0].mean)
            assert torch.allclose(z_comb.logvar, Z[0].logvar)
        else:
            assert torch.allclose(z_comb, Z[0])
    else:
        assert z_comb.shape[1] == len_keep_shared

    if is_variational:
        assert all([isinstance(z, DiagonalGaussian) for z in Z])
        assert isinstance(z_comb, DiagonalGaussian)

@pytest.mark.parametrize("unimodal, is_variational", ([False, False], [True, False], [False, True], [True, True]))
def test_model_forward(unimodal, is_variational):
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if unimodal:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device)]
        model = DisMAE(**unimodal_model_params).to(x_device)
    else:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device), torch.clamp(torch.randn(1, 18, img_size, img_size), 0, 1).to(x_device)]
        model = DisMAE(**model_params).to(x_device)

    W, Z, z_comb, X, masks, q_loss, min_encs = model.forward(data, random_z=True)

    print(f"W shapes: {[w.shape for w in W]}")
    print(f"Z shapes: {[z.shape for z in Z]}")
    print(f"z_comb shape: {z_comb.shape}")
    print(f"X shapes: {[x.shape for x in X]}")
    print(f"q_loss shape: {q_loss.shape}")
    print(f"q_loss: {q_loss}")
    print(f"min_encs shapes: {[me.shape for me in min_encs]}")

    assert all([x.shape == data[i].shape for i, x in enumerate(X)])

@pytest.mark.parametrize("unimodal, is_variational", ([False, False], [True, False], [False, True], [True, True]))
def test_objective(unimodal, is_variational):
    x_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if unimodal:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device)]
        model = MMVAEDataParallelWrapper(DisMAE(**unimodal_model_params).to(x_device))
        objective = DisentanglingObjectiveMAE(model, x_device, **unimodal_objective_params)
    else:
        data = [torch.randn(1, 3, img_size, img_size).to(x_device), (torch.randn(1, 18, img_size, img_size) > 0).to(dtype=torch.float32, device=x_device)]
        model = MMVAEDataParallelWrapper(DisMAE(**model_params).to(x_device))
        objective = DisentanglingObjectiveMAE(model, x_device, **objective_params)

    wlw, ld, mo = objective(model, data, model_params=dict(total_mask_ratio=0.5, shared_mask_ratio=0.5, random_z=True))

    print(f"wlw shape: {wlw.shape}")
    print(f"loss dict: {ld}")