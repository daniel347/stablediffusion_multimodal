import torch
import glob
from .dataset_RGBSemseg import RGBSemseg
from pathlib import Path
import cv2
import numpy as np
import random
from .register_datasets import register_dataset
import math


@register_dataset("mm_ffhq")
class MMFFHQ(RGBSemseg):

    def __init__(self,
                 root_data_dir,
                 transform,
                 split,
                 modality,
                 p_hor_flip=0.0,
                 return_index=False,
                 bisenet_compat=False,
                 single_label=False,
                 **kwargs):

        assert single_label, "FFHQ dataset only supports single label currently"
        super().__init__(transform, modality, False, single_label, bisenet_compat)

        self.root_data_dir = root_data_dir
        self.split = split
        self.return_index = return_index

        self.colour_path = Path(root_data_dir) / "colour"
        self.semseg_path = Path(root_data_dir) / "semseg"

        colour_imgs = glob.glob(str(self.colour_path) + "/**/*.png")
        self.index_list = [int(Path(img).parts[-1].split(".")[0]) for img in colour_imgs]

        if self.split == "train":
            self.index_list = list(filter(lambda x: x < 50000, self.index_list))
        elif self.split == "val":
            self.index_list = list(filter(lambda x: x >= 50000 and x < 60000, self.index_list))
        elif self.split == "test":
            self.index_list = list(filter(lambda x: x >= 60000, self.index_list))

        self.transform = transform
        self.p_hor_flip = p_hor_flip

    def read_colour(self, index):
        img = cv2.imread(str(self.colour_path / f"{((index // 1000) * 1000):05d}/{index:05d}.png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255  # Normalise
        return torch.from_numpy(img).permute((2, 0, 1))

    def read_semseg(self, index):
        semseg = cv2.imread(str(self.semseg_path / f"{((index // 1000) * 1000):05d}/{index:05d}.png"), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(semseg)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        index = self.index_list[idx]

        outs = []
        for mod in self.modality:
            img = self.mode_readers[mod](index)
            if isinstance(self.transform, dict) and mod in self.transform:
                # modality specific transform
                img = self.transform[mod](img)
            elif self.transform is not None:
                # Shared transform for all modalities
                img = self.transform(img)
            outs.append(img)

        flip = random.random() < self.p_hor_flip
        if flip:
            outs = self.flip_augment(outs)

        if self.return_index:
            return outs, index
        else:
            return outs






