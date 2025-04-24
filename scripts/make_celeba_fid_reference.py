import pyiqa
import torch
from pyiqa.archs.fid_arch import generate_reference_statistics
import glob
import os
from ldm.data.dataset_CelebAMask import MMCelebA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = MMCelebA(root_data_dir="../../datasets/CelebAMask-HQ",
                   split='val', transform=None,
                   modality=["colour", "semseg"])


metric = pyiqa.create_metric("kid", device=device)
colour_imgs = [os.path.join(dataset.img_dir, f"{index}.jpg") for index in dataset.index_list]

print("Found {} colour images".format(len(colour_imgs)))

generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="mm_celeba_colour",
                              split="val",
                              model=metric.net.model,
                              metric="KID",
                              device=device,
                              center_crop=True)

# Test computation of KID
test_path = "/home/ds3119/multimodal_generation/outputs/MM_CelebA_Stable_Diffusion/checkpoints/2025_04_05__11_38_31_nvidia_ema_test/temp_fid_colour"
kid = metric(test_path, dataset_name="mm_celeba_colour", dataset_split="val",  dataset_res=299, center_crop=True)

print(f"Computed kid of: {kid}")

