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
            latent_out = {k: v.to("cpu") for k, v in latent.items() if k != "class"}
        elif isinstance(latent, torch.Tensor):
            latent_out = latent.to("cpu")


        if not self.return_class and not self.return_index:
            return latent_out
        else:
            rets = [latent_out]
            if self.return_class:
                rets.append(latent["class"])
            if self.return_index:
                rets.append(idx)

            return rets

