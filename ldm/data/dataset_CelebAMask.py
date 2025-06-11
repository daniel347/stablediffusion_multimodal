import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import torch
from torchvision.transforms.v2.functional import horizontal_flip
from torchvision.transforms.v2 import ColorJitter
import torchvision.utils as tvu
import random
from .register_datasets import register_dataset
from BiSeNetFace.remap_classes import bisenet_to_mm_inds
import glob
import plotly.express as px
import tqdm
import matplotlib.pyplot as plt
import math


@register_dataset("mm_celeba")
class MMCelebA(Dataset):

    def __init__(self,
                 root_data_dir,
                 transform,
                 split,
                 modality,
                 p_hor_flip=0,
                 p_colour_jitter=0,
                 return_index=False,
                 force_reload=False,
                 bisenet_compat=False,
                 single_label=False,
                 **kwargs):

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

        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.transform = transform

        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

        self.return_index = return_index
        self.force_reload = force_reload
        self.bisenet_compat = bisenet_compat or single_label
        self.single_label = single_label

        self.mode_readers = {"colour": self.read_colour, "semseg": self.read_semseg}

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

        if self.single_label:
            combined_png_path = os.path.join(self.mask_dir, f"{index:05d}_combined.png")
            if os.path.exists(combined_png_path) and not self.force_reload:
                # print("Loading precomputed mask")
                try:
                    mask = cv2.imread(combined_png_path, cv2.IMREAD_GRAYSCALE)
                    return torch.from_numpy(mask)
                except Exception:
                    print(f"Failed to load mask {combined_png_path}, reloading")
        else:
            combined_path = os.path.join(self.mask_dir, f"{index:05d}_combined.pt")
            if os.path.exists(combined_path) and not self.force_reload:
                # print("Loading precomputed mask")
                try:
                    return torch.load(combined_path).type(torch.float32)
                except Exception:
                    print(f"Failed to load mask {combined_path}, reloading")

        p = os.path.join(self.mask_dir, f"{index:05d}_*.png")
        mask_files = glob.glob(p)
        mask = np.zeros((len(self.semseg_labels), 512, 512), dtype=np.float32)
        for mask_file in mask_files:
            submask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask_class = "_".join(os.path.basename(mask_file).split(".")[0].split("_")[1:])
            if mask_class not in self.semseg_labels:
                print(f"Error: key {mask_class} not in semseg labels. loading from {mask_file}")
                continue
            mask[self.semseg_labels[mask_class], :, :] = submask // 255

        torch_mask = torch.from_numpy(mask)

        if self.single_label:
            label_mask = self.mask_to_single_labels(torch_mask)
            tvu.save_image(label_mask.unsqueeze(0).float() / 255.0, combined_png_path)
            return label_mask
        else:
            torch.save(torch_mask.type(torch.uint8), combined_path)
            return torch_mask

    def read_colour(self, index):
        img = cv2.imread(os.path.join(self.img_dir, f"{index}.jpg"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255  # Normalise
        return torch.from_numpy(img).permute((2, 0, 1))

    def hex_to_rgb(self, hex):
        hex_str = hex.lstrip("#")
        return torch.Tensor([int(hex_str[i:i+2], 16) for i in [0, 2, 4]])/255.0

    @staticmethod
    def mask_to_single_labels(mask):
        mask_inds = torch.zeros(*mask.shape[1:], dtype=torch.uint8) + 18

        for i in bisenet_to_mm_inds.values():
            if i == 18:  # Background class, which we have automatically filled
                continue
            mask_inds = torch.where(mask[i, :, :] > 0.5, i, mask_inds)
        return mask_inds

    def semseg_display(self, mask):
        if len(mask.shape) == 2:
            # We have a mask of class indices
            mask = self.index_semseg_to_one_hot(mask, num_classes=len(self.semseg_labels))

        mask_rgb = torch.zeros(*(3, *mask.shape[1:]), dtype=torch.float32)
        if self.bisenet_compat:
            # We force the mask to be single class, based on class priority
            mask_inds = self.mask_to_single_labels(mask)

            semseg_display_colours_t = torch.stack([col for col in self.semseg_display_colours.values()] + [torch.zeros(3)])
            mask_rgb = semseg_display_colours_t[mask_inds.to(int)]
            return mask_rgb.permute((2, 0, 1))
        else:
            # Norm so that multiple classes mix colours - We clamp to allow background to be 0 everywhere.
            mask_norm = mask/torch.clamp(mask.sum(dim=0, keepdim=True), 1.0)
            mask_norm = torch.nan_to_num(mask_norm)

            for i in range(mask.shape[0]):
                mask_rgb += self.semseg_display_colours[i][:, None, None] * mask_norm[i, :, :].unsqueeze(0).expand(3, -1, -1).float()
            return mask_rgb

    def colour_display(self, mask):
        return mask

    def create_semseg_colour_index(self):
        colour_dict = self.semseg_display_colours
        class_labels = self.semseg_labels

        num_classes = len(class_labels)
        num_cols = 4

        half_gap = 0.1

        fig, ax = plt.subplots(figsize=(2 * num_cols, math.ceil(num_classes/num_cols)))

        for i, (class_name, class_num) in enumerate(class_labels.items()):
            colour = colour_dict.get(class_num, torch.tensor([1, 1, 1])).numpy()  # Default to white if not found
            colour = tuple([c for c in colour])

            rect_pos = ((i % num_cols) * 2 + half_gap, - (i // num_cols) - half_gap)

            ax.add_patch(plt.Rectangle(rect_pos, 1 - 2*half_gap, 1 - 2*half_gap, color=colour))
            ax.text(rect_pos[0] + 1, rect_pos[1] + 0.5 - half_gap, f"{class_name}",
                    ha='left', va='center', fontsize=10)

        ax.set_xlim(0, 2 * num_cols)
        ax.set_ylim(1, -math.ceil(num_classes/num_cols))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        ax.set_title("Semantic Segmentation Class Legend")

        return fig, ax

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

            # swap left an right channels in semseg
            if "semseg" in self.modality:
                semseg_ind = self.modality.index("semseg")
                semseg = outs[semseg_ind]
                # swap 5 and 6 (eyes), 8 and 9 (brows) and 12 and 13 (ears)
                permutation = [0, 1, 2, 3, 4, 6, 5, 7, 9, 8, 10, 13, 12, 14, 15, 16, 17]
                semseg = semseg.permute(permutation)
                outs[semseg_ind] = semseg

        colour_jitter = random.random() < self.p_colour_jitter
        if colour_jitter and "colour" in self.modality:
            colour_ind = self.modality.index("colour")
            outs[colour_ind] = self.colour_jitter(outs[colour_ind])

        if self.return_index:
            return outs, idx

        return outs
