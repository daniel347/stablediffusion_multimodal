from ldm.data import get_dataset
from torch.utils.data.distributed import DistributedSampler
import torch
from omegaconf import OmegaConf, DictConfig
import torchvision.transforms as v2
import hydra
import pickle
import tqdm

def wrapper(gen):
  while True:
    try:
      yield next(gen)
    except StopIteration:
      break
    except Exception as e:
      print(e) # or whatever kind of logging you want


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

    bad_train_inds = []
    bad_test_inds = []

    for i in tqdm.tqdm(range(len(train_dataset))):
        try:
            # Get the data
            imgs = train_dataset[i]
        except Exception as e:
            print(f"Could not load train image {i}, error: {e}")
            bad_train_inds.append(i)

    for i in tqdm.tqdm(range(len(test_dataset))):
        try:
            # Get the data
            imgs = test_dataset[i]
        except Exception as e:
            print(f"Could not load test image {i}, error: {e}")
            bad_test_inds.append(i)

    bad_inds = {"test": bad_test_inds, "train": bad_train_inds}
    with open("bad_inds.pkl", "wb") as f:
        pickle.dump(bad_inds, f)


if __name__ == "__main__":
    main()