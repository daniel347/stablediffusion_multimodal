from torch.utils.data import Dataset
import plotly.express as px
import tqdm
import torch
from torchvision.transforms import v2
from BiSeNetFace.remap_classes import bisenet_to_mm_inds
from torchvision.transforms.v2.functional import horizontal_flip
from abc import ABC, abstractmethod
import math
import matplotlib.pyplot as plt


class RGBSemseg(ABC, Dataset):
    """This is an abstract class as a base for RGB and semseg modalities.
    It handles conversion of semseg between index, onehot and multi-label forms,
     as well as displaying it as an RGB image. The image loading and iteration are left
     for specific implementations to handle."""

    def __init__(self,
                 transform,
                 modality,
                 force_reload=False,
                 single_label=False,
                 bisenet_compat=False,
                 **kwargs):

        self.modality = modality

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
        self.semseg_display_colours = {num: self.hex_to_rgb(px_colours[i]) for i, num in
                                       enumerate(self.semseg_labels.values())}

        self.mode_readers = {"colour": self.read_colour, "semseg": self.read_semseg}

        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.transform = transform

        self.force_reload = force_reload
        self.bisenet_compat = bisenet_compat or single_label
        self.single_label = single_label

    @abstractmethod
    def read_colour(self, index):
        pass

    @abstractmethod
    def read_semseg(self, index):
        pass

    @abstractmethod
    def __len__(self):
        return None

    @abstractmethod
    def __getitem__(self, idx):
        return None

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

    def flip_augment(self, outs):
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

        return outs

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

    def compute_semseg_class_freqs(self, n_imgs=-1):
        cropped_list = self.index_list[:n_imgs] if n_imgs > 0 else self.index_list
        class_counts = torch.zeros(len(self.semseg_labels))
        for index in tqdm.tqdm(cropped_list):
            img = self.mode_readers["semseg"](index)
            img_counts = img.sum(dim=[1,2])
            class_counts += img_counts

        return class_counts/class_counts.sum()

    @staticmethod
    def hex_to_rgb(hex):
        hex_str = hex.lstrip("#")
        return torch.Tensor([int(hex_str[i:i+2], 16) for i in [0, 2, 4]])/255.0

    @staticmethod
    def one_hot_semseg_to_index(one_hot):
        return torch.argmax(one_hot, dim=-3)

    @staticmethod
    def mask_to_single_labels(mask):
        mask_inds = torch.zeros(*mask.shape[1:], dtype=torch.uint8) + 18

        for i in bisenet_to_mm_inds.values():
            if i == 18:  # Background class, which we have automatically filled
                continue
            mask_inds = torch.where(mask[i, :, :] > 0.5, i, mask_inds)
        return mask_inds

    @staticmethod
    def index_semseg_to_one_hot(index, num_classes=18):
        new_shape = list(index.shape)
        new_shape.insert(-2, num_classes)
        one_hot = torch.zeros(*new_shape, dtype=torch.float32)
        for i in range(num_classes):
            one_hot[..., i, :, :] = (index == i).float()
        return one_hot


def resize_semseg(semseg, shape):
    if len(semseg.shape) == 2:
        semseg = semseg.unsqueeze(0)
        semseg = v2.functional.resize(semseg, shape, v2.InterpolationMode.NEAREST_EXACT, antialias=True).squeeze(0)
    elif len(semseg.shape) == 3:
        semseg = v2.functional.resize(semseg, shape, v2.InterpolationMode.BICUBIC, antialias=True).squeeze(0)
        semseg = torch.clamp(semseg, 0, 1)

    return semseg