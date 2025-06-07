import torch
import glob
import tqdm

model_base_dir = "/rds/general/user/ds3119/ephemeral/outputs/NYU_Depth_Stable_Diffusion/checkpoints/2025_06_05__18_00_06_new_ae_train_lsun/model_*.pt"  
model_paths = glob.glob(model_base_dir)

#print(model_paths)

test_param_name = "model.input_blocks.4.0.in_layers.0.weight"
test_params = []

for model in tqdm.tqdm(model_paths):
    print(f"loading model {model}")
    save = torch.load(model, map_location="cpu")
    curr_param = save["model"][test_param_name]
    test_params.append(curr_param)

    for i, ema in enumerate(save["ema"]["emas"]):
        test_params.append(ema[test_param_name])
        if torch.allclose(ema[test_param_name].data, curr_param.data):
            print(f"error! ema {i} of model {model} is identical to the latest weights")
        for j in range(0, i):
            if torch.allclose(ema[test_param_name].data, test_params[j-i].data):
                print(f"error! ema {i} of model {model} is identical to ema {j}")

for i, p1 in enumerate(test_params):
    for j, p2 in enumerate(test_params):
        if torch.allclose(p1.data, p2.data) and i != j:
            print(f"error! test param {i} == test param {j}")
            

print("All tensors are different")


