import pyiqa
import torch
from pyiqa.archs.fid_arch import generate_reference_statistics
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fid = pyiqa.create_metric("fid", device=device)

nyu_dir = "../../../datasets/nyuv2_processed"
lsun_dir = "../../../datasets/lsun"

colour_imgs = glob.glob(nyu_dir+"/colour/**/*") + glob.glob(lsun_dir+"/colour/**/*")
depth_imgs = glob.glob(lsun_dir+"/depth/**/*") + glob.glob(lsun_dir+"/depth/**/*")

print("Found {} colour images and {} depth images".format(len(colour_imgs), len(depth_imgs)))

"""
generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="nyu_depth_lsun_colour",
                              model=fid.net.model,
                              device=device,
                              center_crop=True)

generate_reference_statistics(dir=None,
                              files=depth_imgs,
                              name="nyu_depth_lsun_depth",
                              model=fid.net.model,
                              device=device,
                              center_crop=True)                    
"""



