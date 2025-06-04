import torch
import glob
import tqdm

model_base_dir = "/home/ds3119/multimodal_generation/outputs/MM_CelebA_Stable_Diffusion/checkpoints/2025_05_02__14_32_30_rgb_sd_ae/*.pt"
model_paths = glob.glob(model_base_dir)

test_param_name = "model.input_blocks.4.0.in_layers.0.weight"
test_params = []

for model in tqdm.tqdm(model_paths):
    save = torch.load(model, map_location="cpu")
    curr_param = save["model"][test_param_name]
    test_params.append(curr_param)

    for i, ema in enumerate(save["ema"]["emas"]):
        test_params.append(ema[test_param_name])
        assert torch.all(ema[test_param_name].data != curr_param.data), f"ema {i} of model {model} is identical to the latest weights"
        for j in range(0, i):
            assert torch.all(ema[test_param_name].data != test_params[j-i].data), f"ema {i} of model {model} is identical to ema {j}"

for p1 in test_params:
    for p2 in test_params:
        assert torch.all(p1.data != p2.data)

print("All tensors are different")


