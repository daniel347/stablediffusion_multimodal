
import pytest
from ldm.data import get_dataset
from torchvision.transforms import v2, InterpolationMode
import torch
import torchvision.utils as tvu
from pathlib import Path

datasets_path = Path(r"C:\Users\danie\Documents\ImperialSoftwareProjects\datasets")

dataset_opts = {"imagenet": {"transform": v2.Compose([v2.Resize(256), v2.CenterCrop(256)]),
                             "modality": ["colour", "depth"],
                             "root_data_dir": datasets_path / "imagenet/ILSVRC/Data/CLS-LOC/",
                             "modality_exts": {"colour": "JPEG", "depth": "png"}},
                "nyu_depth": {"transform": v2.Compose([v2.Resize(256), v2.CenterCrop(256)]),
                              "modality": ["colour", "depth"],
                              "root_data_dir": datasets_path / "nyuv2_processed"},
                "lsun": {"transform": v2.Compose([v2.Resize(256), v2.CenterCrop(256)]),
                              "modality": ["colour", "depth"],
                              "modality_exts": {"colour": "jpg", "depth": "png"},
                              "root_data_dir": datasets_path / "lsun"},
                "mm_celeba": {"transform": {"colour": v2.Resize([256, 256]), "semseg": v2.Resize(256, interpolation=InterpolationMode.NEAREST_EXACT)},
                              "modality": ["colour", "semseg"],
                              "root_data_dir": datasets_path / "CelebAMask-HQ",
                              "single_label": False},
                 "web_lsun": {"transform": {"colour":  v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(256), v2.CenterCrop(256)]),
                                            "depth": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(256), v2.CenterCrop(256)])},
                              "modality": ["colour", "depth"],
                              "root_data_dir": datasets_path / "lsun_web_small"},
                 "web_imagenet": {"transform": {"colour": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(256), v2.CenterCrop(256)]),
                                           "depth": v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(256), v2.CenterCrop(256)])},
                              "modality": ["colour", "depth"],
                              "class_map_path": "LOC_synset_mapping.txt",
                              "root_data_dir": datasets_path / "imagenet_web_small"}}

def get_data_sample(dataset, index=0):
    if not getattr(dataset, "is_iterable", False):
        return dataset[index]
    else:
        return next(iter(dataset))

def unbatch(data):
    if isinstance(data, (list, tuple)):
        return [unbatch(d) for d in data]
    elif isinstance(data, dict):
        return {k: unbatch(v) for k, v in data.items()}
    else:
        return data[0]


@pytest.mark.parametrize("split,dataset_name", [("train", "nyu_depth"), ("train", "imagenet"), ("train", "mm_celeba"), ("train", "lsun"), ("train", "web_lsun"), ("train", "web_imagenet"),
                                                ("test", "nyu_depth"), ("val", "imagenet"), ("val", "mm_celeba"), ("val", "lsun"), ("validation", "web_lsun"), ("validation", "web_imagenet")])
def test_image_load(split, dataset_name):
    loader_name = dataset_name if dataset_name != "lsun" else "nyu_depth"
    if "web" in loader_name:
        loader_name = "webdataset"
    train_dataset = get_dataset(loader_name, **dataset_opts[dataset_name], split=split)
    rgb, depth = get_data_sample(train_dataset)

    # NB depth is actually semseg for mm_celeba
    *_, rgb_h, rgb_w = rgb.shape
    *_, depth_h, depth_w = depth.shape

    assert rgb_w > 0
    assert rgb_h > 0

    assert depth_h > 0
    assert depth_w > 0


@pytest.mark.parametrize("split,dataset_name", [("train", "nyu_depth"), ("train", "imagenet"), ("train", "lsun"), ("train", "web_lsun"), ("train", "web_imagenet"),
                                                ("val", "nyu_depth"), ("val", "imagenet"), ("val", "lsun"), ("validation", "web_lsun"), ("validation", "web_imagenet")])
def test_transform(split, dataset_name):
    loader_name = dataset_name if dataset_name != "lsun" else "nyu_depth"
    if "web" in loader_name:
        loader_name = "webdataset"
    train_dataset = get_dataset(loader_name, **dataset_opts[dataset_name], split=split)
    rgb, depth = get_data_sample(train_dataset)

    assert rgb.shape[-3:] == (3, 256, 256)
    assert depth.shape[-3:] == (1, 256, 256)

    assert rgb.dtype == torch.float32
    assert depth.dtype == torch.float32

    assert torch.allclose(rgb.min(), torch.tensor(0.0))
    assert torch.allclose(rgb.max(), torch.tensor(1.0))

    assert torch.allclose(depth.min(),torch.tensor(0.0))
    assert torch.allclose(depth.max(), torch.tensor(1.0))

@pytest.mark.parametrize("split,dataset_name", [("train", "nyu_depth"), ("train", "lsun"),
                                                ("val", "nyu_depth"), ("val", "lsun")])
def test_class_selection(split, dataset_name):
    loader_name = dataset_name if dataset_name != "lsun" else "nyu_depth"
    dataset = get_dataset(loader_name, **dataset_opts[dataset_name], split=split, classes=["bedroom", "basement"])

    df = dataset.index
    assert (df.colour_path.apply(str).str.contains("bedroom") | df.colour_path.apply(str).str.contains("basement")).all()
    assert df.scene_class.isin(["bedroom", "basement"]).all()


@pytest.mark.parametrize("split,dataset_name,expected_len", [("train", "nyu_depth", 50688), ("train", "imagenet", 1300), ("train", "mm_celeba", 24183), ("train", "lsun", 90000),
                                                             ("val", "nyu_depth", 654), ("val", "imagenet", 50), ("val", "mm_celeba", 2993), ("val", "lsun", 10000)])
def test_len(split, dataset_name, expected_len):
    loader_name = dataset_name if dataset_name != "lsun" else "nyu_depth"
    train_dataset = get_dataset(loader_name, **dataset_opts[dataset_name], split=split)

    assert len(train_dataset) == expected_len


@pytest.mark.parametrize("split,dataset_name", [("train", "mm_celeba"),
                                                ("val", "mm_celeba"),])
def test_celeba_transform(split, dataset_name):
    opts = dataset_opts[dataset_name]
    train_dataset = get_dataset(dataset_name, **opts, split=split)
    rgb, semseg = train_dataset[0]

    assert semseg.shape[1:] == (256, 256)
    assert semseg.dtype == torch.float32
    # One hot vector so this holds
    assert semseg.min() >= torch.tensor(0.0)
    assert semseg.max() <= torch.tensor(1.0)

    assert rgb.shape == (3, 256, 256)
    assert rgb.dtype == torch.float32
    assert rgb.min() >= torch.tensor(0.0)
    assert rgb.max() <= torch.tensor(1.0)


@pytest.mark.parametrize("split,dataset_name", [("train", "mm_celeba"), ("val", "mm_celeba")])
def test_semseg_to_rgb_display(split, dataset_name):
    train_dataset = get_dataset(dataset_name, **dataset_opts[dataset_name], split=split)
    _, semseg = train_dataset[0]

    rgb_display = train_dataset.semseg_display(semseg)
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

def test_celeba_multiple_splits():
    dataset = get_dataset("mm_celeba", **dataset_opts["mm_celeba"], split=["train", "test"])


def test_imagenet_return_class():
    opts = dataset_opts["imagenet"]
    opts["return_class"] = True
    dataset = get_dataset("imagenet", **opts, split="train")

    for i, (data, classes) in enumerate(dataset):
        print(classes)
        print(data[0].shape)

        if i >= 3:
            break

@pytest.mark.parametrize("batched", [True, False])
def test_webdataset_class(batched):
    opts = dataset_opts["web_imagenet"]
    opts["return_class"] = True
    train_dataset = get_dataset("webdataset", **opts, split="train")
    if batched:
        train_dataset.batch_output(4)
    (rgb, depth), cls = get_data_sample(train_dataset)

    if batched:
        assert cls.dtype == torch.int64
        assert len(cls.shape) == 1
    else:
        assert isinstance(cls, int)

@pytest.mark.parametrize("split,dataset_name", [("train", "web_lsun"), ("train", "web_imagenet"),
                                                ("validation", "web_lsun"), ("validation", "web_imagenet")])
def test_webdataset_dataloader(split, dataset_name):
    from torch.utils.data import DataLoader
    loader_name = dataset_name if dataset_name != "lsun" else "nyu_depth"
    if "web" in loader_name:
        loader_name = "webdataset"

    opts = dataset_opts[dataset_name]

    train_dataset = get_dataset(loader_name, **opts, split=split)
    train_dataset.batch_output(4)
    dataloader = DataLoader(train_dataset, batch_size=None, num_workers=0)

    for i, data in enumerate(dataloader):
        rgb, depth = data
        print(rgb)
        assert rgb.shape == (4, 3, 256, 256)
        assert depth.shape == (4, 1, 256, 256)

        if i >= 2:
            break