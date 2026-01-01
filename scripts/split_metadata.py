import hydra
from omegaconf import DictConfig
from pathlib import Path
import jsonlines

@hydra.main(version_base=None, config_path="../configs/data", config_name="lsun_local")
def main(data_conf: DictConfig) -> None:
    root = Path(data_conf.root_data_dir) / "train"

    meta_chunk = []
    chunksize = 16384

    chunk_ind = 0
    chunk_folder = root.parent / f"train_{chunk_ind}"
    chunk_folder.mkdir(exist_ok=True, parents=True)

    with jsonlines.open(root / "metadata.jsonl") as full_meta:
        for i, row in enumerate(full_meta):
            if i % chunksize == 0 and i > 0:
                outpath = chunk_folder / f"metadata.jsonl"
                print(f"saving chunk to {outpath}")
                with jsonlines.open(outpath, "w") as f:
                    f.write_all(meta_chunk)
                meta_chunk = []
                chunk_ind += 1
                chunk_folder = root / f"../train_{chunk_ind}"
                chunk_folder.mkdir(exist_ok=True, parents=True)

            row["colour_file_name"] = str(chunk_folder / f"../train" / row["colour_file_name"])
            row["depth_file_name"] = str(chunk_folder / f"../train" / row["depth_file_name"])
            meta_chunk.append(row)

        if len(meta_chunk) > 0:
            outpath = chunk_folder / f"metadata.jsonl"
            print(f"saving final chunk to {outpath}")
            with jsonlines.open(outpath, "w") as f:
                f.write_all(meta_chunk)

if __name__ == "__main__":
    main()
