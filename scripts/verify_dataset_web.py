import hydra
from omegaconf import OmegaConf, DictConfig
import copy
from ldm.data import get_dataset
from pathlib import Path
import jsonlines
import webdataset as wds
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import InterpolationMode
import torchvision.utils as tvu
import torch
import io
from PIL import Image

@hydra.main(version_base=None, config_path="../configs/data", config_name="lsun_local")
def main(data_conf: DictConfig) -> None:
    root = data_conf.root_data_dir

    tx = {"colour": v2.Compose([v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)]),
          "semseg": v2.Compose([v2.Resize(data_conf.input_size, interpolation=InterpolationMode.NEAREST_EXACT),
                                v2.CenterCrop(data_conf.input_size)]),
          "depth": v2.Compose([v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)])}

    hf_tx = {"colour":  v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)]),
          # "semseg": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(data_conf.input_size, interpolation=InterpolationMode.NEAREST_EXACT),
          #                      v2.CenterCrop(data_conf.input_size)]),
          "depth": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)])}

    hf_decoders = {"colour": lambda x: Image.open(io.BytesIO(x)),
                    "depth": lambda x: Image.open(io.BytesIO(x))}

    def hf_transform(data):
        for mod in data_conf.modality:
            data[mod] = hf_tx[mod](data[mod])
        return data

    return_class = "return_class" in data_conf and data_conf.return_class

    opts = copy.deepcopy(OmegaConf.to_container(data_conf))
    for key in ["p_hor_flip", "p_colour_jitter"]:
        opts.pop(key, None)

    for split in ["train"]:
        hf_dataset = wds.WebDataset("/rds/general/user/ds3119/ephemeral/imagenet_web/train/train-00000.tar").decode(
                        wds.handle_extension("colour.jpg", hf_decoders["colour"]),
                        wds.handle_extension("depth.png", hf_decoders["depth"])).rename(colour="colour.jpg", depth="depth.png").map_dict(**hf_tx)        
        

        for i, hfdata in enumerate(hf_dataset):
            print(f"checking datum {i}")

            print(hfdata["colour"].shape)
            print(hfdata["depth"].shape)
            tvu.save_image(hfdata["colour"], f"test_colour_{i}.png")
            tvu.save_image(hfdata["depth"], f"test_depth_{i}.png")

            if i > 100:
                print("First 100 images checked, stopping.")
                break


if __name__ == "__main__":
    main()
