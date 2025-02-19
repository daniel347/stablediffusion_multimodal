import io
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision import io
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms.v2.functional import horizontal_flip
from torchvision.transforms.v2 import ColorJitter
import random
import glob
from .__init__ import register_dataset


@register_dataset("imagenet")
class ImagenetMultiModal(Dataset):

    def __init__(self, root_data_dir, transform, split, modality, p_hor_flip=0, p_colour_jitter=0):

        super().__init__()
        self.split = split
        self.root_data_dir = root_data_dir

        self.mode_image_paths = {}
        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality

        base_paths = glob.glob(os.path.join(root_data_dir, f"{split}/{modality[0]}/**/*.JPEG"), recursive=True)
        base_mod = modality[0]
        self.mode_image_paths[modality[0]] = base_paths

        for mod in modality[1:]:
            self.mode_image_paths[mod] = [bp.replace(base_mod, mod) for bp in base_paths]

        self.mode_types = {"colour": cv2.IMREAD_COLOR, "depth": cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH}
        self.image_depths = {"colour": 255, "depth": 255} if self.split == "train" else {"colour": 255, "depth": 10000}

        self.transform = transform

        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, idx):
        filepaths = {mod: self.mode_image_paths[idx] for mod in self.modality}

        outs = []
        for mod in self.modality:
            img = cv2.imread(os.path.join(self.root_data_dir, filepaths[mod]), self.mode_types[mod])
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

        return outs
