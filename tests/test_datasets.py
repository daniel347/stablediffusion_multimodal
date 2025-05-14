import pytest
from ldm.data import get_dataset
from ldm.data.dataset_RGBSemseg import resize_semseg
from torchvision.transforms import v2
import torch
import torchvision.utils as tvu

dataset_opts = {"imagenet": {"transform": v2.Compose([v2.Resize(256), v2.CenterCrop(256)]),
                             "modality": ["colour", "depth"],
                             "root_data_dir": "tests/imagenet_mock"},
                "nyu_depth": {"transform": v2.Resize([256, 256]),
                              "modality": ["colour", "depth"],
                              "root_data_dir": "../../datasets/nyuv2_processed"},
                "mm_celeba": {"transform": {"colour": v2.Resize([256, 256]), "semseg": lambda x: resize_semseg(x, (256, 256))},
                              "modality": ["colour", "semseg"],
                              "root_data_dir": "../../datasets/CelebAMask-HQ",
                              "single_label": False},
                "mm_ffhq": {"transform": {"colour": v2.Resize([256, 256]), "semseg": lambda x: resize_semseg(x, (256, 256))},
                            "root_data_dir": "../../datasets/FFHQ_256_Mask",
                            "modality": ["colour", "semseg"],
                            "single_label": True,
                            "return_index": False}}


@pytest.mark.parametrize("split,dataset_name", [("train", "nyu_depth"), ("train", "imagenet"), ("train", "mm_celeba"), ("train", "mm_ffhq"),
                                                ("test", "nyu_depth"), ("val", "imagenet"), ("val", "mm_celeba"), ("val", "mm_ffhq")])
def test_image_load(split, dataset_name):

    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    rgb, depth = train_dataset[0]

    # NB depth is actually semseg for mm_celeba
    _, rgb_h, rgb_w = rgb.shape
    *_, depth_h, depth_w = depth.shape

    assert rgb_w > 0
    assert rgb_h > 0

    assert depth_h > 0
    assert depth_w > 0


@pytest.mark.parametrize("split,dataset_name", [("train", "nyu_depth"), ("train", "imagenet"), ("test", "nyu_depth"), ("val", "imagenet")])
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


@pytest.mark.parametrize("split,dataset_name,single_label", [("train", "mm_celeba", False), ("train", "mm_celeba", True), ("train", "mm_ffhq", True),
                                                             ("val", "mm_celeba", False), ("val", "mm_celeba", True), ("val", "mm_ffhq", True)])
def test_celeba_transform(split, dataset_name, single_label):
    opts = dataset_opts[dataset_name]
    opts["single_label"] = single_label
    train_dataset = get_dataset(dataset_name, **opts, split=split)
    rgb, semseg = train_dataset[0]

    if single_label:
        assert semseg.shape == (256, 256)  # single label by default
        assert semseg.dtype == torch.uint8

        assert semseg.min() >= 0
        assert semseg.max() <= 18
    else:
        assert semseg.shape[1:] == (256, 256)
        assert semseg.dtype == torch.float32
        # One hot vector so this holds
        assert semseg.min() >= 0
        assert semseg.max() <= 1

    assert rgb.shape == (3, 256, 256)
    assert rgb.dtype == torch.float32
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


@pytest.mark.parametrize("split,dataset_name", [("train", "mm_celeba"), ("train", "mm_ffhq"), ("val", "mm_celeba"), ("val", "mm_ffhq")])
def test_semseg_to_rgb_display(split, dataset_name):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    _, semseg = train_dataset[0]

    rgb_display = train_dataset.semseg_display(semseg)
    assert rgb_display.shape == (3, 256, 256)

    assert rgb_display.dtype == torch.float32
    assert rgb_display.min() >= 0.0
    assert rgb_display.max() <= 1.0


@pytest.mark.parametrize("split,dataset_name,expected_len", [("train", "nyu_depth", 50688), ("train", "imagenet", 1300), ("train", "mm_celeba", 24183), ("train", "mm_ffhq", 50000),
                                                             ("test", "nyu_depth", 654), ("val", "imagenet", 50), ("val", "mm_celeba", 2993), ("val", "mm_ffhq", 10000)])
def test_len(split, dataset_name, expected_len):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)

    assert len(train_dataset) == expected_len


def test_celeba_simgle_label_compat_semseg():
    opts = dataset_opts["mm_celeba"]
    opts["single_label"] = True
    dataset = get_dataset("mm_celeba", **opts, split="train")
    assert dataset.single_label == True

    _, semseg = dataset[0]

    assert semseg.shape == (256, 256)
    assert semseg.min() >= 0
    assert semseg.max() <= 18

    rgb_display = dataset.semseg_display(semseg)
    assert rgb_display.shape == (3, 256, 256)
    assert rgb_display.dtype == torch.float32
    assert rgb_display.min() >= 0.0
    assert rgb_display.max() <= 1.0


def test_celeba_single_label_vis_images():
    opts = dataset_opts["mm_celeba"]
    opts["single_label"] = True
    dataset = get_dataset("mm_celeba", **dataset_opts["mm_celeba"], split="train")

    for i, (rgb, semseg) in enumerate(dataset):
        semseg_disp = dataset.semseg_display(semseg)
        tvu.save_image(semseg_disp, f"celeba_semseg_{i}.png")

        if i >= 3:
            break


