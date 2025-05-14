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
from .register_datasets import register_dataset
from pathlib import Path
import glob


@register_dataset("nyu_depth")
class NYUDepth(Dataset):

    def __init__(self, root_data_dir, transform, split, modality, p_hor_flip=0, p_colour_jitter=0, return_index=False, **kwargs):

        super().__init__()
        self.split = split
        self.root_data_dir = root_data_dir

        self.scene_colour_paths = list((Path(root_data_dir) / "colour").iterdir())
        self.scene_depth_paths = list((Path(root_data_dir) / "depth").iterdir())
        self.scenes = [scene.name for scene in self.scene_colour_paths]

        self.scenes = pd.DataFrame({"scene_class": [s.rsplit("_", 1)[0] for s in self.scenes],
                                   "scene_ind": [s.rsplit("_", 1)[1] for s in self.scenes],
                                   "scene_colour_path": self.scene_colour_paths,
                                   "scene_depth_path": self.scene_depth_paths})

        self.scenes = self.scenes.sort_values(by=["scene_class", "scene_ind"])

        if self.split == "test":
            self.scenes = self.scenes.groupby("scene_class", as_index=False).last()
        elif self.split == "train":
            self.scenes = self.scenes.groupby("scene_class", as_index=False).apply(lambda x: x.iloc[:-1])
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.index = self.scenes.apply(self.make_scene_dataframe, axis=1)

        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.transform = transform

        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

        self.return_index = return_index

        self.mode_readers = {"colour": self.read_colour, "depth": self.read_depth}

    def make_scene_dataframe(self, x):
        colour_paths = list(x.scene_colour_path.iterdir())
        depth_paths = list(x.scene_depth_path.iterdir())

        # Get only the files with the same index
        colour_inds = [int(p.name.split(".")[0]) for p in colour_paths]
        depth_inds = [int(p.name.split(".")[0]) for p in depth_paths]

        combined_inds = list(set(colour_inds).union(depth_inds))
        colour_paths = [x.scene_colour_path / f"{i}.png" for i in combined_inds]
        depth_paths = [x.scene_depth_path / f"{i}.png" for i in combined_inds]

        assert len(colour_paths) == len(depth_paths), f"paths {x.scene_colour_path} and {x.scene_depth_path} have different numbers of files!"
        scene_class = [x.scene_class for _ in colour_paths]
        scene_ind = [x.scene_ind for _ in colour_paths]

        return pd.DataFrame({"colour_path": colour_paths, "depth_path": depth_paths, "scene_class": scene_class, "scene_ind": scene_ind})

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, idx):
        filepaths = self.index.iloc[idx]

        outs = []
        for mod in self.modality:
            img = self.mode_readers[mod](str(filepaths[f"{mod}_path"]))
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

        if self.return_index:
            return outs, idx
        else:
            return outs

    def depth_display(self, img):
        return img.repeat(3, 1, 1)

    def colour_display(self, img):
        return img

    def read_colour(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        return torch.from_numpy(img).permute((2, 0, 1))

    def read_depth(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 65535
        return torch.from_numpy(img).permute((2, 0, 1))

