import pyiqa
import torch
from pyiqa.archs.fid_arch import generate_reference_statistics
import glob
from torchvision.transforms import v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fid = pyiqa.create_metric("fid", device=device)

nyu_dir = "../../../datasets/nyuv2_processed"
lsun_dir = "../../../datasets/lsun"

colour_imgs = glob.glob(nyu_dir+"/colour/bedroom_*/*") + glob.glob(lsun_dir+"/colour/**/*")
depth_imgs = glob.glob(nyu_dir+"/depth/bedroom_*/*") + glob.glob(lsun_dir+"/depth/**/*")

print("Found {} colour images and {} depth images".format(len(colour_imgs), len(depth_imgs)))

fid = pyiqa.create_metric("fid", device=device)

generate_reference_statistics(dir=None,
                              files=depth_imgs,
                              name="nyu_depth_lsun_colour",
                              model=fid.net.model,
                              device=device,
                              split="val",
                              center_crop=True,
                              res=128,
                              intermediate_resize=v2.Resize((128, 128))
                              )

generate_reference_statistics(dir=None,
                              files=depth_imgs,
                              name="nyu_depth_lsun_depth",
                              model=fid.net.model,
                              device=device,
                              split="val",
                              center_crop=True,
                              res=128,
                              intermediate_resize=v2.Resize((128, 128))
                              )


kid = pyiqa.create_metric("kid", device=device)

generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="nyu_depth_lsun_colour",
                              model=kid.net.model,
                              device=device,
                              split="val",
                              metric="KID",
                              center_crop=True,
                              res=128,
                              intermediate_resize=v2.Resize((128, 128))
                              )

generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="nyu_depth_lsun_depth",
                              model=kid.net.model,
                              device=device,
                              split="val",
                              metric="KID",
                              center_crop=True,
                              res=128,
                              intermediate_resize=v2.Resize((128, 128))
                              )