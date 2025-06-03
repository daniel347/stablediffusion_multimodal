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
import pickle
from abc import ABC, abstractmethod


class RGBDepth(ABC, Dataset):

    def __init__(self, transform, modality, return_index=False, **kwargs):

        super().__init__()
        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.transform = transform

        self.return_index = return_index

        self.mode_readers = {"colour": self.read_colour, "depth": self.read_depth}

    @abstractmethod
    def __len__(self):
        return None

    @abstractmethod
    def __getitem__(self, idx):
        return None

    def depth_display(self, img):
        return img.repeat(3, 1, 1)

    def colour_display(self, img):
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

    def save_bad_paths(self, bad_paths):
        with (Path(self.root_data_dir) / f"bad_paths_{self.split}.pkl").open("wb") as bpf:
            pickle.dump(bad_paths, bpf)


if __name__ == "__main__":
    train_dataset = NYUDepth("../../datasets/nyuv2_processed", None, "test", ["colour", "depth"], 0, 0, False)
    rgb, depth = train_dataset[0]

    # NB depth is actually semseg for mm_celeba
    _, rgb_h, rgb_w = rgb.shape
    *_, depth_h, depth_w = depth.shape

    assert rgb_w > 0
    assert rgb_h > 0

    assert depth_h > 0
    assert depth_w > 0