from ldm.data import get_dataset
from torchvision.transforms import v2
import torch
import time
import tqdm

dataset_opts = {"mm_celeba": {"transform": None, # v2.Compose([v2.Resize([256, 256])]),
                              "modality": ["colour", "semseg"],
                              "root_data_dir": "../../datasets/CelebAMask-HQ"}}

def check_celeba_loading_speeds(train_loader, device="cuda:1"):
    start = time.time()

    for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        pass

    end = time.time()
    print(f"Iterated through dataset in {end - start} seconds")

def preload_dataset(split="test", dataset_name="mm_celeba", modality=["colour", "semseg"]):
    dataset_opts["modality"] = modality
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    kwargs = {'num_workers': 0, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

    return train_loader


if __name__ == "__main__":
    train_loader = preload_dataset("train")
    check_celeba_loading_speeds(train_loader)

    test_loader = preload_dataset("test")
    check_celeba_loading_speeds(test_loader)
