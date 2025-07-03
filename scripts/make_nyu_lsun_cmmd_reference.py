import torch
import cmmd_pytorch as cmmd
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nyu_dir = "../../../datasets/nyuv2_processed"
lsun_dir = "../../../datasets/lsun"

colour_imgs = glob.glob(nyu_dir+"/colour/bedroom_0001a/*") + glob.glob(lsun_dir+"/colour/bedroom_0/*")
depth_imgs = glob.glob(nyu_dir+"/depth/bedroom_0001a/*") + glob.glob(lsun_dir+"/depth/bedroom_0/*")

print("Found {} colour images and {} depth images".format(len(colour_imgs), len(depth_imgs)))

cmmd.compute_and_cache_embeddings(dir=None,
                                  files=colour_imgs,
                                  dataset_name="nyu_depth_lsun_colour",
                                  dataset_res=299,
                                  dataset_split="val",
                                  device=device)

cmmd.compute_and_cache_embeddings(dir=None,
                                  files=depth_imgs,
                                  dataset_name="nyu_depth_lsun_depth",
                                  dataset_res=299,
                                  dataset_split="val",
                                  device=device)

# Test computation of CMMD
test_path = "/rds/general/user/ds3119/ephemeral/outputs/NYU_Depth_Stable_Diffusion/checkpoints/2025_05_29__12_29_12_new_ae_train/temp_fid_colour"
cmmd_val = cmmd.compute_cmmd(test_path, dataset_name="nyu_depth_lsun_colour", dataset_split="val",  dataset_res=299, device=device)

print(f"Computed colour CMMD of: {cmmd_val}")

test_path = "/rds/general/user/ds3119/ephemeral/outputs/NYU_Depth_Stable_Diffusion/checkpoints/2025_05_29__12_29_12_new_ae_train/temp_fid_depth"
cmmd_val = cmmd.compute_cmmd(test_path, dataset_name="nyu_depth_lsun_depth", dataset_split="val",  dataset_res=299, device=device)

print(f"Computed depth CMMD of: {cmmd_val}")
