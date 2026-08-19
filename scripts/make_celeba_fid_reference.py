import pyiqa
import torch
from pyiqa.archs.fid_arch import generate_reference_statistics
import os
from ldm.data.dataset_CelebAMask import MMCelebA
from torchvision.transforms import v2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = MMCelebA(root_data_dir="../../datasets/CelebAMask-HQ",
                   split='all', transform=None,
                   modality=["colour", "semseg"])


colour_imgs = [os.path.join(dataset.img_dir, f"{index}.jpg") for index in dataset.index_list]

print("Found {} colour images".format(len(colour_imgs)))


metric = pyiqa.create_metric("fid", device=device)
generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="mm_celeba_colour",
                              split="all",
                              model=metric.net.model,
                              # metric="KID",
                              device=device,
                              center_crop=True,
                              res=128,
                              intermediate_resize=v2.Resize((128, 128))
                              )

# Test computation of FID
test_path = "/path/to/example/images"
fid = metric(test_path, dataset_name="mm_celeba_colour", dataset_split="all",  dataset_res=128, center_crop=True)

print(f"Computed fid of: {fid}")


metric = pyiqa.create_metric("kid", device=device)
generate_reference_statistics(dir=None,
                              files=colour_imgs,
                              name="mm_celeba_colour",
                              split="all",
                              model=metric.net.model,
                              metric="KID",
                              device=device,
                              center_crop=True,
                              res=128,
                              intermediate_resize=v2.Resize((128, 128))
                              )

# Test computation of KID
test_path = "/path/to/example/images"
kid = metric(test_path, dataset_name="mm_celeba_colour", dataset_split="all",  dataset_res=128, center_crop=True)

print(f"Computed kid of: {kid}")
