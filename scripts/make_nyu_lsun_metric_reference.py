import pyiqa
import torch
from .fid_arch_pyiqa import generate_reference_statistics
import cmmd_pytorch as cmmd
import glob

'''
This script generates reference statistics for the NYU depth and LSUN combined dataset for FID, KID, and CMMD metrics. 
Reference files are saved to the scripts folder and should be moved to the cache folder within your pyiqa installation 
directory.
'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fid = pyiqa.create_metric("fid", device=device)

nyu_dir = "../../datasets/nyuv2_processed"
lsun_dir = "../../datasets/lsun"

colour_imgs = glob.glob(nyu_dir+"/colour/bedroom_*/*") + glob.glob(lsun_dir+"/colour/**/*")
depth_imgs = glob.glob(nyu_dir+"/depth/bedroom_*/*") + glob.glob(lsun_dir+"/depth/**/*")

print("Found {} colour images and {} depth images".format(len(colour_imgs), len(depth_imgs)))

"""
FID Reference
"""
fid = pyiqa.create_metric("fid", device=device)

generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="nyu_depth_lsun_new_colour",
                              model=fid.net.model,
                              device=device,
                              split="val",
                              center_crop=True,
                              res=299,
                              # intermediate_resize=v2.Resize((128, 128))
                              )

generate_reference_statistics(dir=None,
                              files=depth_imgs,
                              name="nyu_depth_lsun_new_depth",
                              model=fid.net.model,
                              device=device,
                              split="val",
                              center_crop=True,
                              res=299,
                              # intermediate_resize=v2.Resize((128, 128))
                              )

"""
KID Reference
"""
kid = pyiqa.create_metric("kid", device=device)

generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="nyu_depth_lsun_new_colour",
                              model=kid.net.model,
                              device=device,
                              split="val",
                              metric="KID",
                              center_crop=True,
                              res=299,
                              # intermediate_resize=v2.Resize((128, 128))
                              )

generate_reference_statistics(dir=None,
                              files=depth_imgs,
                              name="nyu_depth_lsun_new_depth",
                              model=kid.net.model,
                              device=device,
                              split="val",
                              metric="KID",
                              center_crop=True,
                              res=299,
                              # intermediate_resize=v2.Resize((128, 128))
                              )

"""
CMMD Reference
"""
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