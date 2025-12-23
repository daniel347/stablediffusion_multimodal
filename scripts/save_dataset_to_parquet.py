import hydra
from omegaconf import OmegaConf, DictConfig
import copy
from pathlib import Path
import math
import datasets

@hydra.main(version_base=None, config_path="../configs/data", config_name="lsun_local")
def main(data_conf: DictConfig) -> None:
    outpath = Path(data_conf.get("root_data_dir", )) / "../parquet_dataset/data"
    outpath.mkdir(parents=True, exist_ok=True)

    opts = copy.deepcopy(OmegaConf.to_container(data_conf))
    for key in ["p_hor_flip", "p_colour_jitter"]:
        opts.pop(key, None)

    for split in ["train", "val"]:

        ds = datasets.load_dataset("imagefolder", data_dir=data_conf.root_data_dir, split="train")
        ds = ds.filter(lambda x: x["split"] == split)
        n_shards = math.ceil(len(ds) / 16384)

        for index in range(n_shards):
            shard = ds.shard(index=index, num_shards=n_shards, contiguous=True)
            output_path_template = outpath / f"{split}-{index:05d}-of-{n_shards:05d}.parquet"
            shard.to_parquet(output_path_template)

if __name__ == "__main__":
    main()