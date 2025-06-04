import torch
import glob
import tqdm

model_base_dir = "/rds/general/user/ds3119/ephemeral/outputs/NYU_Depth_Stable_Diffusion/checkpoints/2025_05_29__12_29_12_new_ae_train/model_*000.pt"  # the 000 is a hack to exclude the EMA output models
model_paths = glob.glob(model_base_dir)

print(model_paths)

test_param_name = "model.input_blocks.4.0.in_layers.0.weight"
test_params = []

for model in tqdm.tqdm(model_paths):
    save = torch.load(model, map_location="cpu")
    curr_param = save["model"][test_param_name]
    test_params.append(curr_param)

    for i, ema in enumerate(save["ema"]["emas"]):
        test_params.append(ema[test_param_name])
        if not torch.all(ema[test_param_name].data != curr_param.data):
            print(f"error! ema {i} of model {model} is identical to the latest weights")
        for j in range(0, i):
            if not torch.all(ema[test_param_name].data != test_params[j-i].data):
                print(f"error! ema {i} of model {model} is identical to ema {j}")

for i, p1 in enumerate(test_params):
    for j, p2 in enumerate(test_params):
        if not torch.all(p1.data != p2.data) and i != j:
            print(f"error! test param {i} == test param {j}")
            

print("All tensors are different")


