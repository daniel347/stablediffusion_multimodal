import pyiqa
import torch
from pyiqa.archs.fid_arch import generate_reference_statistics
import glob
import os
from ldm.data.dataset_CelebAMask import MMCelebA

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = MMCelebA(root_data_dir="../mmvaeplus/data/CelebAMask-HQ",
                   split='val', transform=None,
                   modality=["colour", "semseg"])


fid = pyiqa.create_metric("fid", device=device)
colour_imgs = [os.path.join(dataset.img_dir, f"{index}.jpg") for index in dataset.index_list]

print("Found {} colour images".format(len(colour_imgs)))

generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="mm_celeba_colour",
                              split="val",
                              model=fid.net.model,
                              device=device,
                              center_crop=True)