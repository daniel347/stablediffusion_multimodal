import pytest
from ldm.data.dataset_NYU_Depth import NYUDepth
from torchvision.transforms import v2
import torch

@pytest.mark.parametrize("train", [False, True])
def test_image_load(train):
    train_dataset = NYUDepth("../mmvaeplus/data/nyu_data", transform=None, train=train, modality=['colour', 'depth'])
    rgb, depth = train_dataset[0]

    _, rgb_h, rgb_w = rgb.shape
    _, depth_h, depth_w = depth.shape

    assert rgb_w > 0
    assert rgb_h > 0

    assert depth_h > 0
    assert depth_w > 0


@pytest.mark.parametrize("train", [False, True])
def test_transform(train):
    tx = v2.Compose([v2.Resize([256, 256]), v2.ToTensor(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = NYUDepth("../mmvaeplus/data/nyu_data", transform=tx, train=train, modality=['colour', 'depth'])
    rgb, depth = train_dataset[0]
    assert rgb.shape == (3, 256, 256)
    assert depth.shape == (1, 256, 256)

    assert rgb.dtype == torch.float32
    assert depth.dtype == torch.float32

    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0

    assert depth.min() >= 0.0
    assert depth.max() <= 1.0


@pytest.mark.parametrize("train, expected_len", [(False, 654), (True, 50688)])
def test_len(train, expected_len):
    tx = v2.Compose([v2.Resize([256, 256]), v2.ToTensor(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = NYUDepth("../mmvaeplus/data/nyu_data", transform=tx, train=train, modality=['colour', 'depth'])

    assert len(train_dataset) == expected_len
