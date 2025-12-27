import hydra
from omegaconf import OmegaConf, DictConfig
import copy
from ldm.data import get_dataset
from pathlib import Path
import jsonlines
import tqdm

@hydra.main(version_base=None, config_path="../configs/data", config_name="lsun_local")
def main(data_conf: DictConfig) -> None:
    root = data_conf.root_data_dir

    tx = None
    return_class = "return_class" in data_conf and data_conf.return_class

    opts = copy.deepcopy(OmegaConf.to_container(data_conf))
    for key in ["p_hor_flip", "p_colour_jitter"]:
        opts.pop(key, None)

    meta = []

    splitnames = {"train": "train", "val": "validation"}

    for split in ["train", "val"]:
        impath_train = Path(root) / splitnames[split]
        outpath = impath_train / f"metadata.jsonl"
        dataset = get_dataset(data_conf.dataset_name,
                                    transform=tx,
                                    split=split,
                                    return_row=True,
                                    return_class=return_class,
                                    dont_load_image=True,
                                    **opts)

        for i, data in tqdm.tqdm(enumerate(dataset)):
            colour_path = Path(data[-1]["colour_path"])
            depth_path = Path(data[-1]["depth_path"])

            colour_new_path = Path(root) / splitnames[split] / colour_path.relative_to(root)
            depth_new_path = Path(root) / splitnames[split] / depth_path.relative_to(root)
            depth_new_path.parent.mkdir(exist_ok=True, parents=True)
            colour_new_path.parent.mkdir(exist_ok=True, parents=True)

            colour_path.rename(colour_new_path)
            depth_path.rename(depth_new_path)

            meta_row = {"colour_file_name": str(colour_new_path.relative_to(impath_train)),
                        "depth_file_name": str(depth_new_path.relative_to(impath_train)), "ind": i, "split": split}

            if return_class:
                meta_row["label"] = data[1]["class"]
            meta.append(meta_row)

        print(f"found: {len(meta)} image pairs")

        with jsonlines.open(outpath, "w") as f:
            f.write_all(meta)

if __name__ == "__main__":
    main()
