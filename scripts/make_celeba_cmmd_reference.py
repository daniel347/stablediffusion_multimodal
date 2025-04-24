import torch
import cmmd_pytorch as cmmd
import glob
import os
from ldm.data.dataset_CelebAMask import MMCelebA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = MMCelebA(root_data_dir="../../datasets/CelebAMask-HQ",
                   split='val', transform=None,
                   modality=["colour", "semseg"])

colour_imgs = [os.path.join(dataset.img_dir, f"{index}.jpg") for index in dataset.index_list]

print("Found {} colour images".format(len(colour_imgs)))

cmmd.compute_and_cache_embeddings(dir=None,
                                  files=colour_imgs,
                                  dataset_name="mm_celeba_colour",
                                  dataset_res=299,
                                  dataset_split="val",
                                  device=device)

# Test computation of CMMD
test_path = "/home/ds3119/multimodal_generation/outputs/MM_CelebA_Stable_Diffusion/checkpoints/2025_04_05__11_38_31_nvidia_ema_test/temp_fid_colour"
cmmd_val = cmmd.compute_cmmd(test_path, dataset_name="mm_celeba_colour", dataset_split="val",  dataset_res=299, device=device)

print(f"Computed CMMD of: {cmmd_val}")
