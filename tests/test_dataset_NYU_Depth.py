import pytest
from ldm.data import get_dataset
from torchvision.transforms import v2
import torch

dataset_opts = {"imagenet": {"transform": v2.Compose([v2.Resize(256), v2.CenterCrop(256)]),
                             "modality": ["colour", "depth"],
                             "root_data_dir": "tests/imagenet_mock"},
                "nyu_depth": {"transform": v2.Compose([v2.Resize([256, 256])]),
                              "modality": ["colour", "depth"],
                              "root_data_dir": "../mmvaeplus/data/nyu_data"},
                "mm_celeba": {"transform": v2.Compose([v2.Resize([256, 256])]),
                              "modality": ["colour", "semseg"],
                              "root_data_dir": "../mmvaeplus/data/CelebAMask-HQ"}}


@pytest.mark.parametrize("split,dataset_name", [("train", "nyu_depth"), ("train", "imagenet"), ("train", "mm_celeba"), ("val", "nyu_depth"), ("val", "imagenet"), ("val", "mm_celeba")])
def test_image_load(split, dataset_name):

    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    rgb, depth = train_dataset[0]

    # NB depth is actually semseg for mm_celeba
    _, rgb_h, rgb_w = rgb.shape
    _, depth_h, depth_w = depth.shape

    assert rgb_w > 0
    assert rgb_h > 0

    assert depth_h > 0
    assert depth_w > 0


@pytest.mark.parametrize("split,dataset_name", [("train", "nyu_depth"), ("train", "imagenet"), ("val", "nyu_depth"), ("val", "imagenet")])
def test_transform(split, dataset_name):

    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    rgb, depth = train_dataset[0]
    assert rgb.shape == (3, 256, 256)
    assert depth.shape == (1, 256, 256)

    assert rgb.dtype == torch.float32
    assert depth.dtype == torch.float32

    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0

    assert depth.min() >= 0.0
    assert depth.max() <= 1.0


@pytest.mark.parametrize("split,dataset_name", [("train", "mm_celeba"), ("val", "mm_celeba")])
def test_celeba_transform(split, dataset_name):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    rgb, semseg = train_dataset[0]
    assert rgb.shape == (3, 256, 256)
    assert semseg.shape[1:] == (256, 256)

    assert rgb.dtype == torch.float32
    assert semseg.dtype == torch.float32

    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0

    # One hot vector so this holds
    assert semseg.min() >= 0
    assert semseg.max() <= 1


@pytest.mark.parametrize("split,dataset_name", [("train", "mm_celeba"), ("val", "mm_celeba")])
def test_semseg_to_rgb_display(split, dataset_name):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    _, semseg = train_dataset[0]

    rgb_display = train_dataset.semseg_display(semseg)
    assert rgb_display.shape == (3, 256, 256)

    assert rgb_display.dtype == torch.float32
    assert rgb_display.min() >= 0.0
    assert rgb_display.max() <= 1.0


@pytest.mark.parametrize("split,dataset_name,expected_len", [("train", "nyu_depth", 50688), ("train", "imagenet", 1300), ("train", "mm_celeba", 24183), ("val", "nyu_depth", 654), ("val", "imagenet", 50), ("val", "mm_celeba", 2993)])
def test_len(split, dataset_name, expected_len):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)

    assert len(train_dataset) == expected_len
