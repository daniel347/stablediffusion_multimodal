import torch
import cmmd_pytorch as cmmd
import glob
import os
from ldm.data.dataset_CelebAMask import MMCelebA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = MMCelebA(root_data_dir="../../datasets/CelebAMask-HQ",
                   split='train', transform=None,
                   modality=["colour", "semseg"])

colour_imgs = [os.path.join(dataset.img_dir, f"{index}.jpg") for index in dataset.index_list]
print("Found {} colour images".format(len(colour_imgs)))

cmmd.compute_and_cache_embeddings(dir=None,
                                  files=colour_imgs,
                                  dataset_name="mm_celeba_colour",
                                  dataset_res=299,
                                  dataset_split="train",
                                  device=device)

# Test computation of CMMD
test_path = "/rds/general/user/ds3119/ephemeral/outputs/MM_CelebA_Stable_Diffusion/checkpoints/2025_07_01__02_38_23_new_hyp_sample_latents/test_fid_colour"
cmmd_val = cmmd.compute_cmmd(test_path, dataset_name="mm_celeba_colour", dataset_split="train",  dataset_res=299, device=device)

print(f"Computed CMMD of: {cmmd_val}")
