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


@register_dataset("nyu_depth")
class NYUDepth(Dataset):

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
                 modality_exts="png",
                 **kwargs):

        super().__init__()
        self.split = split
        self.root_data_dir = root_data_dir

        if isinstance(modality, str):
            modality = [modality]
        self.modality = modality
        self.modality_exts = modality_exts if isinstance(modality_exts, dict) else {m: modality_exts for m in modality}
        self.transform = transform

        self.scene_colour_paths = list((Path(root_data_dir) / "colour").iterdir())
        self.scene_depth_paths = [Path(str(p).replace("colour", "depth")) for p in self.scene_colour_paths]
        self.scenes = [scene.name for scene in self.scene_colour_paths]

        self.scenes = pd.DataFrame({"scene_class": [s.rsplit("_", 1)[0] for s in self.scenes],
                                   "scene_ind": [s.rsplit("_", 1)[1] for s in self.scenes],
                                   "scene_colour_path": self.scene_colour_paths,
                                   "scene_depth_path": self.scene_depth_paths})

        self.scenes = self.scenes.sort_values(by=["scene_class", "scene_ind"])

        if self.split == "test" or self.split == "val":
            self.scenes = self.scenes.groupby("scene_class", as_index=False).last()
        elif self.split == "train":
            self.scenes = self.scenes.groupby("scene_class", as_index=False).apply(lambda x: x.iloc[:-1])
        elif self.split == "all":
            pass
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if classes is not None:
            # Restrict to just the provided classes
            self.scenes = self.scenes[self.scenes.scene_class.isin(classes)]

        index_dfs = []
        for _, scene in self.scenes.iterrows():
            index_dfs.append(self.make_scene_dataframe(scene))
        self.index = pd.concat(index_dfs, axis=0, ignore_index=True)



        self.p_hor_flip = p_hor_flip
        self.p_colour_jitter = p_colour_jitter
        if p_colour_jitter > 0:
            self.colour_jitter = ColorJitter(0.1, 0.1, 0.1, 0.02)

        self.return_index = return_index
        self.return_row = return_row

        self.mode_readers = {"colour": self.read_colour, "depth": self.read_depth}

        self.bad_paths = []  # record the indices that dont load properly
        try:
            with (Path(root_data_dir) / f"bad_paths_{split}.pkl").open("rb") as bpf:
                self.bad_paths = pickle.load(bpf)
        except FileNotFoundError:
            print("Could not load bad paths")
            pass

        # add other modalities of the same index
        self.bad_paths += [(p.replace("colour", "depth") if "colour" in p else p.replace("depth", "colour")) for p in self.bad_paths]

        # Convert relative to absolute paths
        self.bad_paths = [Path(self.root_data_dir) / p for p in self.bad_paths]

        # Remove the bad paths from the index
        for m in modality:
            self.index = self.index[~self.index[f"{m}_path"].isin(self.bad_paths)]

    def make_scene_dataframe(self, x):
        paths = []
        inds = []
        for m in self.modality:
            paths.append(list(x[f"scene_{m}_path"].iterdir()))
            inds.append(set([int(p.name.split(".")[0]) for p in paths[-1]]))

        combined_inds = list(set(inds[0]).intersection(*inds[1:]))

        combined_paths = []
        for m in self.modality:
            combined_paths.append([x[f"scene_{m}_path"] / f"{i}.{self.modality_exts[m]}" for i in combined_inds])

        assert all([len(cp) == len(combined_paths[0]) for cp in combined_paths]), f"paths have different numbers of files!"

        scene_class = [x.scene_class for _ in combined_paths[0]]
        scene_ind = [x.scene_ind for _ in combined_paths[0]]

        out_dict = {"scene_class": scene_class, "scene_ind": scene_ind}
        for i, m in enumerate(self.modality):
            out_dict[f"{m}_path"] = combined_paths[i]

        return pd.DataFrame(out_dict)

    def __len__(self):
        return len(self.index.index)

    def __getitem__(self, idx):
        filepaths = self.index.iloc[idx]

        outs = []
        for mod in self.modality:
            img = self.mode_readers[mod](str(filepaths[f"{mod}_path"]))
            if img is None:
                # Return the filepath so we can add it to bad_paths
                return str(filepaths[f"{mod}_path"].relative_to(self.root_data_dir))
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
            return outs, filepaths
        else:
            return outs

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
    train_dataset = NYUDepth("../../datasets/lsun", None, "train", ["colour", "depth"], 0, 0, False, modality_exts={"colour": "jpg", "depth": "png"})
    rgb, depth = train_dataset[0]

    # NB depth is actually semseg for mm_celeba
    _, rgb_h, rgb_w = rgb.shape
    *_, depth_h, depth_w = depth.shape

    assert rgb_w > 0
    assert rgb_h > 0

    assert depth_h > 0
    assert depth_w > 0
