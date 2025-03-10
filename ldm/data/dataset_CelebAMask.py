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
import glob
import plotly.express as px
import tqdm


@register_dataset("mm_celeba")
class MMCelebA(Dataset):

    def __init__(self, root_data_dir, transform, split, modality, p_hor_flip=0, p_colour_jitter=0, return_index=False, **kwargs):

        super().__init__()
        self.split = split
        self.root_data_dir = root_data_dir

        self.img_dir = os.path.join(root_data_dir, "CelebA-HQ-img")
        self.mask_dir = os.path.join(root_data_dir, "CelebAMask-HQ-mask-anno")

        img_list = glob.glob(f"{self.img_dir}/*.jpg")
        self.index_list = [int(os.path.basename(img).split(".")[0]) for img in img_list]
        self.semseg_labels = {"skin": 0,
                                "nose": 1,
                                "l_lip": 2,
                                "u_lip": 3,
                                "hair": 4,
                                "r_eye": 5,
                                "l_eye": 6,
                                "neck": 7,
                                "l_brow": 8,
                                "r_brow": 9,
                                "mouth": 10,
                                "cloth": 11,
                                "l_ear": 12,
                                "r_ear": 13,
                                "ear_r": 14,
                                "neck_l": 15,
                                "eye_g": 16,
                                "hat": 17}

        px_colours = px.colors.qualitative.Dark24
        self.semseg_display_colours = {num: self.hex_to_rgb(px_colours[i]) for i, num in enumerate(self.semseg_labels.values())}

        # Select the data split according to the original CelebA split
        self.partition_file = pd.read_csv(os.path.join(root_data_dir, "list_eval_partition.txt"), delim_whitespace=True, header=None, names=["orig_idx", "split"])
        self.partition_file.orig_idx = self.partition_file.orig_idx.apply(lambda x: int(x.split(".")[0]))
        self.index_mapping = pd.read_csv(os.path.join(root_data_dir, "CelebA-HQ-to-CelebA-mapping.txt"), delim_whitespace=True)
        self.index_mapping.index = self.index_mapping.orig_idx

        self.partition_file.index = self.partition_file.orig_idx
        self.partition_hq = self.partition_file.loc[self.index_mapping.orig_idx]
        self.index_mapping["split"] = self.partition_hq.split

        self.split_num = {"train": 0, "val": 1, "test": 2}[self.split]

        self.index_list = self.index_mapping[self.index_mapping.split == self.split_num].idx.values
        print(f"Selected split {self.split} and found {len(self.index_list)} images")

        self.mode_readers = {"colour": self.read_colour, "semseg": self.read_semseg}

        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.transform = transform

        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

        self.return_index = return_index

    def make_semseg_class_list(self, index_list):
        self.semseg_labels = {}
        for idx in tqdm.tqdm(index_list[:500]):
            p = os.path.join(self.mask_dir, f"{idx:05d}_*.png")
            mask_files = glob.glob(p)
            mask_classes = ["_".join(os.path.basename(mask_file).split(".")[0].split("_")[1:]) for mask_file in mask_files]
            for mask_class in mask_classes:
                if mask_class not in self.semseg_labels:
                    self.semseg_labels[mask_class] = len(self.semseg_labels)

        print("Found the following labels: ", self.semseg_labels)

    def compute_semseg_class_freqs(self, n_imgs=-1):
        cropped_list = self.index_list[:n_imgs] if n_imgs > 0 else self.index_list
        class_counts = torch.zeros(len(self.semseg_labels))
        for index in tqdm.tqdm(cropped_list):
            img = self.mode_readers["semseg"](index)
            img_counts = img.sum(dim=[1,2])
            class_counts += img_counts

        return class_counts/class_counts.sum()

    def read_semseg(self, index):
        p = os.path.join(self.mask_dir, f"{index:05d}_*.png")
        mask_files = glob.glob(p)
        mask = np.zeros((len(self.semseg_labels), 512, 512), dtype=np.float32)
        for mask_file in mask_files:
            submask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask_class = "_".join(os.path.basename(mask_file).split(".")[0].split("_")[1:])
            mask[self.semseg_labels[mask_class], :, :] = submask // 255

        return torch.from_numpy(mask)

    def read_colour(self, index):
        img = cv2.imread(os.path.join(self.img_dir, f"{index}.jpg"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255  # Normalise
        return torch.from_numpy(img).permute((2, 0, 1))

    def hex_to_rgb(self, hex):
        hex_str = hex.lstrip("#")
        return torch.Tensor([int(hex_str[i:i+2], 16) for i in [0, 2, 4]])/255.0

    def semseg_display(self, mask):
        mask_rgb = torch.zeros(*(3, *mask.shape[1:]), dtype=torch.float32)
        # Norm so that multiple classes mix colours - We clamp to allow background to be 0 everywhere.
        mask_norm = mask/torch.clamp(mask.sum(dim=0, keepdim=True), 1.0)
        mask_norm = torch.nan_to_num(mask_norm)
        for i in range(mask.shape[0]):
            mask_rgb += self.semseg_display_colours[i][:, None, None] * mask_norm[i, :, :].unsqueeze(0).expand(3, -1, -1).float()
        return mask_rgb

    def colour_display(self, mask):
        return mask

    @staticmethod
    def one_hot_semseg_to_index(one_hot):
        return torch.argmax(one_hot, dim=-3)

    @staticmethod
    def index_semseg_to_one_hot(index, num_classes=18):
        new_shape = list(index.shape)
        new_shape.insert(-2, num_classes)
        one_hot = torch.zeros(*new_shape, dtype=torch.float32)
        for i in range(num_classes):
            one_hot[..., i, :, :] = (index == i).float()
        return one_hot

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
            for i in range(len(outs)):
                outs[i] = horizontal_flip(outs[i])

        colour_jitter = random.random() < self.p_colour_jitter
        if colour_jitter and "colour" in self.modality:
            colour_ind = self.modality.index("colour")
            outs[colour_ind] = self.colour_jitter(outs[colour_ind])

        if self.return_index:
            return outs, idx

        return outs
