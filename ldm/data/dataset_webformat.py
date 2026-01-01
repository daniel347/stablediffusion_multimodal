import io
from torch.utils.data import IterableDataset
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms.v2.functional import horizontal_flip
import random
from .register_datasets import register_dataset
from pathlib import Path
import webdataset as wds


@register_dataset("webdataset")
class WebDataset(IterableDataset):

    def __init__(self,
                 root_data_dir,
                 transform,
                 split,
                 modality,
                 p_hor_flip=0,
                 return_class=False,
                 class_map_path=None,
                 return_index=False,
                 return_row=False,
                 **kwargs):

        super().__init__()
        if isinstance(split, list):
            split = split[0]
        self.split = split
        self.root_data_dir = root_data_dir

        self.mode_image_paths = {}
        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.transform = transform if transform is not None else {m: lambda x: x for m in modality}

        data_dir = Path(root_data_dir) / split
        data_nums = [int(d.name.split(".")[0].split("-")[-1]) for d in data_dir.glob("*.tar")]
        data_num_min, data_num_max = min(data_nums), max(data_nums)
        wds_path = "train-{%s..%s}.tar" % (str(data_num_min).zfill(5), str(data_num_max).zfill(5))
        wds_path = str(data_dir / wds_path).replace("\\", "/")

        self.ds = (wds.WebDataset(wds_path)
                   .shuffle(1000 if split == "train" else 0)
                   .decode(wds.handle_extension("colour.jpg", self.colour_decoder),
                          wds.handle_extension("depth.png", self.depth_decoder))
                   .rename(colour="colour.jpg", depth="depth.png")
                   .map_dict(**self.transform)
                   .map(self.map_output))

        self.return_class = return_class
        if self.return_class and class_map_path:
            classes = []
            with open(Path(root_data_dir) / class_map_path) as f:
                for ind, line in enumerate(f):
                    synset = line.split(" ")[0].strip("-n")
                    name = " ".join(line.split(" ")[1:]).split(",")[0]  # Take first name only
                    classes.append({"num": ind, "synset": synset, "name": name})

            self.classes = pd.DataFrame(classes)
            self.classes.index = self.classes["synset"]

        self.p_hor_flip = p_hor_flip

        self.return_index = return_index
        self.return_row = return_row
        self.is_iterable = True

    @staticmethod
    def colour_decoder(x):
        return Image.open(io.BytesIO(x))

    @staticmethod
    def depth_decoder(x):
        return Image.open(io.BytesIO(x))

    def batch_output(self, B):
        self.ds = self.ds.batched(B).map(self.make_tensors)

    def class_num_to_name(self, class_num):
        if not hasattr(self, "classes"):
            raise ValueError("Can't fetch class name without a class mapping")
        return self.classes[self.classes.num == class_num].iloc[0].name

    def __iter__(self):
        return iter(self.ds)

    def map_output(self, sample):
        outs = [sample[m] for m in self.modality]

        flip = random.random() < self.p_hor_flip
        if flip:
            for i in range(len(outs)):
                outs[i] = horizontal_flip(outs[i])

        if not self.return_class and not self.return_index and not self.return_row:
            return outs
        else:
            rets = [outs]
            if self.return_class:
                class_code = sample["__key__"].split("_")[0].lstrip("n")
                if not hasattr(self, "classes"):
                    rets.append(int(class_code))
                else:
                    class_num = self.classes.loc[class_code, "num"]
                    rets.append(int(class_num))
            if self.return_index:
                raise ValueError("return_index not supported in WebDataset iterator")
            elif self.return_row:
                fp_dict = {"path": sample["__key__"]}
                rets.append(fp_dict)

            return rets

    def make_tensors(self, data):
        imgs = data if not (self.return_class or self.return_row) else data[0]
        if isinstance(imgs, list):
            imgs = [torch.stack([s[m] for s in imgs]) for m in range(len(imgs[0]))]

        if self.return_class or self.return_row:
            data_out = [imgs]
            if self.return_class:
                data_out.append(torch.tensor(data[1], dtype=torch.long))
            if self.return_row:
                data_out.append(data[-1])
            return tuple(data_out)
        return imgs

    def depth_display(self, img, opencv=False):
        print("Depth img min: ", img.min().item(), "max:", img.max().item())
        if opencv:
            return (torch.clamp(img, 0, 1) * 65535).to(torch.uint16)[0].numpy()
        else:
            return img.repeat(3, 1, 1)

    def colour_display(self, img, opencv=False):
        print("Colour img min: ", img.min().item(), "max:", img.max().item())
        if opencv:
            return (torch.clamp(img, 0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()[:, :, ::-1]
        else:
            return img

if __name__ == "__main__":
    from torchvision.transforms import v2
    opts = {"transform": {"colour":  v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(128), v2.CenterCrop(128)]),
                                            "depth": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(128), v2.CenterCrop(128)])},
                              "modality": ["colour", "depth"],
                              "root_data_dir": "../../datasets/lsun_web_small"},
    dataset = WebDataset(**opts)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

    for i, data in enumerate(dataloader):
        colours, depths, class_nums = data
        print("Batch ", i, " colours shape: ", colours.shape, " depths shape: ", depths.shape,
              " class nums: ", class_nums)
        if i == 2:
            break