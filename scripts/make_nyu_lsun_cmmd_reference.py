import torch
import cmmd_pytorch as cmmd
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nyu_dir = "../../../datasets/nyuv2_processed"
lsun_dir = "../../../datasets/lsun"

colour_imgs = glob.glob(nyu_dir+"/colour/bedroom_*/*") + glob.glob(lsun_dir+"/colour/bedroom_*/*")
depth_imgs = glob.glob(nyu_dir+"/depth/bedroom_*/*") + glob.glob(lsun_dir+"/depth/bedroom_*/*")

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
test_path = "path/to/example/images"
cmmd_val = cmmd.compute_cmmd(test_path, dataset_name="nyu_depth_lsun_colour", dataset_split="val",  dataset_res=299, device=device)

# print(f"Computed colour CMMD of: {cmmd_val}")

test_path = "path/to/example/images"
cmmd_val = cmmd.compute_cmmd(test_path, dataset_name="nyu_depth_lsun_depth", dataset_split="val",  dataset_res=299, device=device)

print(f"Computed depth CMMD of: {cmmd_val}")
