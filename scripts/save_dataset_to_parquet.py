import hydra
from omegaconf import OmegaConf, DictConfig
import copy
from pathlib import Path
import math
import datasets
import tqdm

@hydra.main(version_base=None, config_path="../configs/data", config_name="lsun_local")
def main(data_conf: DictConfig) -> None:
    outpath = Path(data_conf.get("root_data_dir", )) / "../lsun_full_parquet/data"
    outpath.mkdir(parents=True, exist_ok=True)

    shards = list(Path(data_conf.root_data_dir).glob("train_*"))
    n_shards = len(shards)
    for index, shard in enumerate(shards):
        ds = datasets.load_dataset("imagefolder", data_dir=shard,
                                   split="train", verification_mode=datasets.VerificationMode.NO_CHECKS)
        output_path_template = outpath / f"train-{index:05d}-of-{n_shards:05d}.parquet"
        ds.to_parquet(output_path_template)


    ds_val = datasets.load_dataset("imagefolder", data_dir=Path(data_conf.root_data_dir) / "validation",
                                   split="train", verification_mode=datasets.VerificationMode.NO_CHECKS)
    ds_val.to_parquet(outpath / f"validation-00000-of-00001.parquet")

if __name__ == "__main__":
    main()
