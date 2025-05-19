from ldm.data import get_dataset
import torch
from omegaconf import OmegaConf, DictConfig
import torchvision.transforms as v2
import hydra
import pickle
import tqdm
from torch.utils.data import DataLoader
import multiprocessing as mp

def safe_get_item(args):
    idx, dataset, split = args
    try:
        ret = dataset[idx]
        if isinstance(ret, str):
            return (split, idx, None, ret)  # No error returned, but a file path
        return None  # No error
    except Exception as e:
        return (split, idx, str(e), None)  # Error info


@hydra.main(version_base=None, config_path="configs/training/", config_name="nyu_depth_autoencoder_train.yaml")
def main(conf: DictConfig) -> None:
    tx = {"colour": v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)]),
          "semseg": v2.Compose([v2.Resize(conf.data.input_size, interpolation=v2.InterpolationMode.NEAREST_EXACT),
                                v2.CenterCrop(conf.data.input_size)]),
          "depth": v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)])}

    train_dataset = get_dataset(conf.data.dataset_name,
                                root_data_dir=conf.data.datadir,
                                transform=tx,
                                split="train",
                                modality=["colour", "depth"],
                                p_hor_flip=0.5,
                                p_colour_jitter=0.1)
    test_dataset = get_dataset(conf.data.dataset_name,
                               root_data_dir=conf.data.datadir,
                               transform=tx,
                               split="test",
                               modality=["colour", "depth"])

    train_args = [(i, train_dataset, "train") for i in range(len(train_dataset))]
    test_args = [(i, test_dataset, "test") for i in range(len(test_dataset))]
    all_args = train_args + test_args

    bad_train_inds = []
    bad_test_inds = []

    bad_train_paths = []
    bad_test_paths = []

    """

    for args in all_args:
        ret = safe_get_item(args)
        if ret is not None:
            split, idx, err = ret
            print(f"Could not load {split} image {idx}, error: {err}")
            if split == "train":
                bad_train_inds.append(idx)
            else:
                bad_test_inds.append(idx)
    """

    with mp.Pool(processes=8) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(safe_get_item, all_args), total=len(all_args)):
            if result is not None:
                split, idx, err, fp = result
                print(f"Could not load {split} image {idx}, error: {err}, path: {fp}")
                if split == "train":
                    bad_train_inds.append(idx)
                    if fp is not None:
                        bad_train_paths.append(fp)
                else:
                    bad_test_inds.append(idx)
                    if fp is not None:
                        bad_test_paths.append(fp)

    print(bad_train_inds)
    print(bad_test_inds)

    print(bad_train_paths)
    print(bad_test_paths)

    train_dataset.save_bad_paths(bad_train_paths)
    test_dataset.save_bad_paths(bad_test_paths)

    bad_inds = {"test": bad_test_inds, "train": bad_train_inds}
    with open("bad_inds.pkl", "wb") as f:
        pickle.dump(bad_inds, f)


if __name__ == "__main__":
    main()