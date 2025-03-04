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


@register_dataset("imagenet")
class ImagenetMultiModal(Dataset):

    def __init__(self, root_data_dir, transform, split, modality, p_hor_flip=0, p_colour_jitter=0, return_class=False):

        super().__init__()
        self.split = split
        self.root_data_dir = root_data_dir

        self.mode_image_paths = {}
        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality

        self.image_extensions = {"colour": "JPEG", "depth": "JPX"}

        base_mod = modality[0]
        base_mod_ext = self.image_extensions[base_mod]
        base_paths = glob.glob(os.path.join(root_data_dir, f"{split}/{base_mod}/**/*.{base_mod_ext}"), recursive=True)

        self.mode_image_paths[modality[0]] = base_paths

        for mod in modality[1:]:
            self.mode_image_paths[mod] = [f"{bp.replace(base_mod, mod).strip(base_mod_ext)}{self.image_extensions[mod]}" for bp in base_paths]

        self.mode_types = {"colour": cv2.IMREAD_COLOR, "depth": cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH}
        self.image_depths = {"colour": 255, "depth": 65535} if self.split == "train" else {"colour": 255, "depth": 65535}

        self.transform = transform

        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

        self.return_class = return_class
        if self.return_classes:
            classes = []
            with open(os.path.join(root_data_dir, "imagenet_classes.txt")) as f:
                json_classes = json.load(f)
                for class_num, info in json_classes.items():
                    classes.append({"num": class_num, "synset": info["id"].rstrip("-n"), "name": info["label"]})

            self.classes = pd.DataFrame(classes)
            self.classes.index = classes["synset"]

    def class_num_to_name(self, class_num):
        return self.classes[self.classes.num == class_num].iloc[0].name

    def __len__(self):
        return len(self.mode_image_paths[self.modality[0]])

    def __getitem__(self, idx):
        filepaths = {mod: self.mode_image_paths[mod][idx] for mod in self.modality}

        outs = []
        for mod in self.modality:
            img = cv2.imread(filepaths[mod], self.mode_types[mod])
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / self.image_depths[mod]  # Normalise
            img = torch.Tensor(img).permute((2, 0, 1))
            if self.transform is not None:
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

        if self.return_class:
            synset = filepaths[self.modality[0]].split("/")[-2].lstrip("n")
            class_num = self.classes.loc[synset, "num"]
            return outs, class_num
        else:
            return outs
