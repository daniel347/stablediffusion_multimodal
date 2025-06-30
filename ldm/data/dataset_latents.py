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
from torch.distributions import Normal
import random
import glob
from .register_datasets import register_dataset
from pathlib import Path


@register_dataset("latents")
class LatentDataset(Dataset):

    def __init__(self,  latent_path, split, modality, return_class=False, return_index=False, **kwargs):
        super().__init__()

        latent_dir = Path(latent_path)
        self.latents = list(latent_dir.glob("**/*.pt"))

        self.modality = modality

        self.return_class = return_class
        if self.return_class:
            self.classes = [int(path.parent.name) for path in self.latents]
        self.return_index = return_index

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = torch.load(self.latents[idx], weights_only=True)
        if isinstance(latent, dict):
            # Return a distribution to support sampling
            latent = {"mean": latent["mean"].to("cpu"), "std": latent["std"].to("cpu")}
        elif isinstance(latent, torch.Tensor):
            latent = latent.to("cpu")

        if not self.return_class and not self.return_index:
            return latent
        else:
            rets = [latent]
            if self.return_class:
                rets.append(self.classes[idx])
            if self.return_index:
                rets.append(idx)

            return rets

"""
    def collate_fn(self, data):
        print(f"collate called with data: {data}")
        if isinstance(data[0], Normal):
            mean = torch.stack([d.mean for d in data])
            std = torch.stack([d.std for d in data])
            collated = Normal(mean, std)
            return collated
        elif isinstance(data[0], torch.Tensor):
            collated = torch.stack(data)
            return collated
        elif isinstance(data[0], list) or isinstance(data[0], tuple):
            # Collate each element of the list
            return (self.collate_fn([d[i] for d in data]) for i in range(len(data[0]))) 
        else:
            return torch.tensor(data)
"""

