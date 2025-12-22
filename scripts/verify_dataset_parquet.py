import hydra
from omegaconf import OmegaConf, DictConfig
import copy
from ldm.data import get_dataset
from pathlib import Path
import jsonlines
from datasets import load_dataset
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import InterpolationMode
import torch

@hydra.main(version_base=None, config_path="../configs/data", config_name="lsun_local")
def main(data_conf: DictConfig) -> None:
    root = data_conf.root_data_dir
    outpath = Path(root) / "metadata.jsonl"

    tx = {"colour": v2.Compose([v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)]),
          "semseg": v2.Compose([v2.Resize(data_conf.input_size, interpolation=InterpolationMode.NEAREST_EXACT),
                                v2.CenterCrop(data_conf.input_size)]),
          "depth": v2.Compose([v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)])}

    hf_tx = {"colour":  v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)]),
          "semseg": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(data_conf.input_size, interpolation=InterpolationMode.NEAREST_EXACT),
                                v2.CenterCrop(data_conf.input_size)]),
          "depth": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)])}

    def hf_transform(data):
        for mod in data_conf.modality:
            data[mod] = hf_tx[mod](data[mod])
        return data

    return_class = "return_class" in data_conf and data_conf.return_class

    opts = copy.deepcopy(OmegaConf.to_container(data_conf))
    for key in ["p_hor_flip", "p_colour_jitter"]:
        opts.pop(key, None)

    for split in ["train", "val"]:
        dataset = get_dataset(data_conf.dataset_name,
                                    transform=tx,
                                    split=split,
                                    return_row=False,
                                    return_class=return_class,
                                    **opts)

        hf_dataset = load_dataset(r"C:/Users/danie/Documents/ImperialSoftwareProjects/datasets/parquet_dataset")
        hf_dataset.set_transform(hf_transform)
        hf_split = hf_dataset[split]

        for i, (hfdata, data) in enumerate(zip(hf_split, dataset)):
            assert torch.allclose(hfdata["colour"], data[0]), f"Colour data mismatch for index {data[-1]['ind']}"
            assert torch.allclose(hfdata["depth"],
                                  data[1]), f"Depth data mismatch for index {data[-1]['ind']}"

            if i > 100:
                print("All data is matching for first 100 samples")
                break

if __name__ == "__main__":
    main()