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

@register_dataset("latents")
class LatentDataset(Dataset):

    def __init__(self,  latent_path, split, modality, return_class=False, return_index=False, **kwargs):
        super().__init__()

        latent_dir = Path(f"{latent_path}_{split}")
        self.latents = list(latent_dir.glob("**/*.pt"))

        self.modality = modality

        self.return_class = return_class
        if self.return_class:
            self.classes = [int(path.parent.name) for path in self.latents]
        self.return_index = return_index

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = torch.load(self.latents[idx], weights_only=True).to("cpu")
        if not self.return_class and not self.return_index:
            return latent
        else:
            rets = [latent]
            if self.return_class:
                rets.append(self.classes[idx])
            if self.return_index:
                rets.append(idx)

            return rets