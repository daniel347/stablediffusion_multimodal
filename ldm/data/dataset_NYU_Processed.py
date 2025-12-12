from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from torchvision.transforms.v2.functional import horizontal_flip
from torchvision.transforms.v2 import ColorJitter
import random
from .register_datasets import register_dataset
import mat73
from pathlib import Path


@register_dataset("nyu_labelled")
class NYULabelled(Dataset):

    def __init__(self,
                 root_data_dir,
                 transform,
                 split,
                 modality,
                 p_hor_flip=0,
                 p_colour_jitter=0,
                 return_index=False,
                 classes=None,
                 return_row=False,
                 **kwargs):

        super().__init__()
        self.split = split
        self.root_data_dir = root_data_dir

        self.data = mat73.loadmat(Path(self.root_data_dir) / "nyu_depth_v2_labeled.mat")
        self.data["colour"] = self.data.pop("images")
        self.data["depth"] = self.data.pop("depths")
        self.data["semseg"] = self.data.pop("labels")

        for k in ["accelData", "namesTolds", "rawDepths", "rawRgbFilenames", "rawDepthFilenames"]:
            self.data.pop(k, None)

        for k in ["names", "sceneTypes", "scenes"]:
            self.data[k] = [n[0] for n in self.data[k]]


        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.transform = transform

        self.mode_readers = {"colour": self.read_colour, "depth": self.read_depth, "semseg": self.read_semseg}

        if classes is not None:
            # Restrict to just the provided classes
            self.filter_data(classes)

        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

        self.return_index = return_index
        self.return_row = return_row


    def filter_data(self, classes):
        inds = [i for i, t in enumerate(self.data["sceneTypes"]) if t in classes]
        self.data["depth"] = self.data["depth"][:, :, inds]
        self.data["colour"] = self.data["colour"][:, :, :, inds]
        self.data["semseg"] = self.data["semseg"][:, :, inds]
        self.data["sceneTypes"] = [self.data["sceneTypes"][i] for i in inds]
        self.data["scenes"] = [self.data["scenes"][i] for i in inds]

    def __len__(self):
        return self.data["colour"].shape[-1]

    def __getitem__(self, idx):

        outs = []
        for mod in self.modality:
            img = self.mode_readers[mod](self.data[mod][..., idx])
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
        elif self.return_row:
            return outs, {}
        else:
            return outs

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

    def semseg_display(self, img, opencv=False):
        # TODO: This is garbage but we dont need it yet
        print("Semseg img min: ", img.min().item(), "max:", img.max().item())
        if opencv:
            return img[0].numpy().astype(np.uint8)
        else:
            return img.repeat(3, 1, 1)

    def read_colour(self, np_img):
        img = np_img.astype(np.float32) / 255
        return torch.from_numpy(img).permute((2, 0, 1))

    def read_depth(self, np_img):
        img = np.clip(np_img, 0, 10.0) / 10.0

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32)
        return torch.from_numpy(img).permute((2, 0, 1))

    def read_semseg(self, np_img):
        img = np_img
        return torch.from_numpy(img).unsqueeze(0)
