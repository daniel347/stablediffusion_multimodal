from torchvision.transforms import v2
import torch
from ldm.data import get_dataset

dataset_opts = {"imagenet": {"transform": v2.Compose([v2.Resize(256), v2.CenterCrop(256)]),
                             "modality": ["colour", "depth"],
                             "root_data_dir": "tests/imagenet_mock"},
                "nyu_depth": {"transform": v2.Compose([v2.Resize([256, 256])]),
                              "modality": ["colour", "depth"],
                              "root_data_dir": "../mmvaeplus/data/nyu_data"}}

train_dataset = get_dataset("imagenet", **dataset_opts["imagenet"], split="train")
test_dataset = get_dataset("imagenet", **dataset_opts["imagenet"], split="val")

print("=== Train dataset ===")
for i, (rgb, depth) in enumerate(train_dataset):
    print(f"depth min: {torch.min(depth)} max: {torch.max(depth)}")
    print(f"rgb min: {torch.min(rgb)} max: {torch.max(rgb)}")

    if i > 5:
        break

print("=== Test dataset ===")
max_depth_val = 0

for i, (rgb, depth) in enumerate(test_dataset):
    print(f"depth min: {torch.min(depth)} max: {torch.max(depth)}")
    print(f"rgb min: {torch.min(rgb)} max: {torch.max(rgb)}")

    if torch.max(depth) > max_depth_val:
        max_depth_val = torch.max(depth)

    if i > 50:
        break

print("max_depth_val", max_depth_val)



