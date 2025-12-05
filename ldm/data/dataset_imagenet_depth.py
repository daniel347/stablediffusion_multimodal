import io
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision import io
from PIL import Image
import cv2
import numpy as np
import json
import torch
from torchvision.transforms.v2.functional import horizontal_flip
from torchvision.transforms.v2 import ColorJitter
import random
import glob
from .register_datasets import register_dataset
from pathlib import Path


@register_dataset("imagenet")
class ImagenetMultiModal(Dataset):

    def __init__(self,
                 root_data_dir,
                 transform,
                 split,
                 modality,
                 p_hor_flip=0,
                 p_colour_jitter=0,
                 return_class=False,
                 return_index=False,
                 return_row=False,
                 modality_exts="JPEG",
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
        self.modality_exts = modality_exts if isinstance(modality_exts, dict) else {m: modality_exts for m in modality}
        self.transform = transform

        base_mod = modality[0]
        base_mod_ext = self.modality_exts[base_mod]
        base_paths = glob.glob(os.path.join(root_data_dir, f"{split}/{base_mod}/**/*.{base_mod_ext}"), recursive=True)

        self.mode_image_paths[modality[0]] = base_paths

        for mod in modality[1:]:
            self.mode_image_paths[mod] = [f"{bp.replace(base_mod, mod).strip(base_mod_ext)}{self.modality_exts[mod]}" for bp in base_paths]

        self.mode_types = {"colour": cv2.IMREAD_COLOR, "depth": cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH}
        self.image_depths = {"colour": 255, "depth": 65535} if self.split == "train" else {"colour": 255, "depth": 65535}

        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

        self.return_class = return_class
        if self.return_class:
            classes = []
            with open(os.path.join(root_data_dir, "imagenet_classes.txt")) as f:
                json_classes = json.load(f)
                for class_num, info in json_classes.items():
                    classes.append({"num": class_num, "synset": info["id"].rstrip("-n"), "name": info["label"]})

            self.classes = pd.DataFrame(classes)
            self.classes.index = classes["synset"]

        self.return_index = return_index
        self.return_row = return_row

        self.mode_readers = {"colour": self.read_colour, "depth": self.read_depth}

    def class_num_to_name(self, class_num):
        return self.classes[self.classes.num == class_num].iloc[0].name

    def __len__(self):
        return len(self.mode_image_paths[self.modality[0]])

    def __getitem__(self, idx):
        filepaths = {mod: self.mode_image_paths[mod][idx] for mod in self.modality}

        outs = []
        for mod in self.modality:
            img = self.mode_readers[mod](str(filepaths[f"{mod}"]))
            if img is None:
                # Return the filepath so we can add it to bad_paths
                return str(filepaths[f"{mod}_path"].relative_to(self.root_data_dir))
            if self.transform is not None:
                if isinstance(self.transform, dict):
                    img = self.transform[mod](img)
                else:
                    img = self.transform(img)

            outs.append(img)

        flip = random.random() < self.p_hor_flip
        if flip:
            for i in range(len(outs)):
                outs[i] = horizontal_flip(outs[i])

        colour_jitter = random.random() < self.p_colour_jitter
        if colour_jitter and "colour" in self.modality:
            colour_ind = self.modality.index("colour")
            outs[colour_ind] = self.colour_jitter(outs[colour_ind])

        if not self.return_class and not self.return_index:
            return outs
        else:
            rets = [outs]
            if self.return_class:
                synset = filepaths[self.modality[0]].split("/")[-2].lstrip("n")
                class_num = self.classes.loc[synset, "num"]
                rets.append(class_num)
            if self.return_index:
                rets.append(idx)
            elif self.return_row:
                fp_dict = filepaths
                for k in fp_dict.keys():
                    if isinstance(fp_dict[k], Path):
                        fp_dict[k] = str(fp_dict[k])
                rets.append(fp_dict)

            return rets

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

    def read_colour(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("Couldnt load rgb image", path)
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        return torch.from_numpy(img).permute((2, 0, 1))

    def read_depth(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if img is None:
            print("Couldnt load depth image", path)
            return None

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 65535
        return torch.from_numpy(img).permute((2, 0, 1))
