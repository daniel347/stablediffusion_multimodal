from ldm.data import get_dataset
from torchvision.transforms import v2
import cv2
import numpy as np
import torch
import time
import tqdm

dataset_opts = {"mm_celeba": {"transform": v2.Compose([v2.Resize([256, 256])]),
                              "modality": ["colour", "semseg"],
                              "root_data_dir": "../../datasets/CelebAMask-HQ"}}


def image_load(split="train", dataset_name="mm_celeba"):

    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    rgb, depth = train_dataset[0]

    # NB depth is actually semseg for mm_celeba
    _, rgb_h, rgb_w = rgb.shape
    _, depth_h, depth_w = depth.shape

    assert rgb_w > 0
    assert rgb_h > 0

    assert depth_h > 0
    assert depth_w > 0

    # Check overlap of semseg classes

    overlap_img = depth.sum(0) > 1
    cv2.imwrite("overlap_img.png", (overlap_img.numpy() * 255).astype("uint8"))
    cv2.imwrite("rgb_img.png", (np.transpose(rgb.numpy(), (1, 2, 0))[:, :, ::-1] * 255).astype("uint8"))

    rgb_display = train_dataset.semseg_display(depth)

    cv2.imwrite("semseg_img.png", (np.transpose(rgb_display.numpy(), (1, 2, 0))[:, :, ::-1] * 255).astype("uint8"))

def compute_class_weights(split="train", dataset_name="mm_celeba"):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    freqs = train_dataset.compute_semseg_class_freqs()
    print(freqs)
    torch.save(freqs, "../mmvaeplus/data/CelebAMask-HQ/semseg_class_freqs.pt")


def draw_semseg_legend(split="train", dataset_name="mm_celeba"):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    fig, ax = train_dataset.create_semseg_colour_index()
    fig.savefig("semseg_legend.png")


def check_celeba_loading_speeds(train_loader, device="cuda:1"):
    start = time.time()

    for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        data = [d.to(device) for d in data]
        print(data[0].shape)
        if i > 10:
            break

    end = time.time()
    print(f"Iterated through dataset in {end - start} seconds")

def preload_dataset(split="test", dataset_name="mm_celeba", modality=["colour", "semseg"]):
    dataset_opts["modality"] = modality
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

    return train_loader


if __name__ == "__main__":
    loader = preload_dataset()
    import cProfile

    cProfile.run("check_celeba_loading_speeds(loader)", "celeba_load_prof.out")

    """
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    check_celeba_loading_speeds()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.dump_stats('load_celeba.prof')
    """

