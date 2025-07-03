# Util functions
import math
import os
import shutil
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from torchvision.utils import make_grid, save_image
from pathlib import Path
import wandb
from omegaconf import OmegaConf, ListConfig, DictConfig
from torchvision.transforms import v2, InterpolationMode
from ldm.data import get_dataset, MMCelebA
import copy
import logging
logger = logging.getLogger(__name__)
from torch.utils.data import DistributedSampler
from image_diversity import ClipMetrics, InceptionMetrics
from pathlib import Path

# Constants
class Constants(object):
    eta = 1e-20


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model_light(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)
    # if hasattr(model, 'vaes'):
        # for vae in model.vaes:
            # fdir, fext = os.path.splitext(filepath)
            # save_vars(vae.state_dict(), fdir + '_' + vae.modelName + fext)

def unpack_data_PM(data, device='cuda'):
    data_nolabel = data[0]
    n_idxs = len(data_nolabel)
    return [data_nolabel[idx].to(device) for idx in range(n_idxs)], data[1].to(device)

def unpack_data_cubIC(data, device='cuda'):
    return [data[0][0].to(device), data[1][0].to(device)]


def unpack_data_nyu(data, device="cuda"):
    return [data[0].to(device), data[1].to(device)]


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


class NonLinearLatent_Classifier(nn.Module):
    """ Non-linear Latent classifier defintion. """

    def __init__(self, in_n, out_n):
        super(NonLinearLatent_Classifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_n, 64), nn.ReLU(inplace=True),
                                 nn.Linear(64, out_n), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.mlp(x)

def get_10_polymnist_samples(polymnist, num_testing_images, device):
    """
    Function to get PolyMNIST samples for qualitative examples of cross-reconstruction at test time
    """
    samples = []
    for i in range(10):
        while True:
            imgs, target = polymnist.__getitem__(random.randint(0, num_testing_images - 1))
            img_mnist, img_svhn, img_3, img_4, img_5 = imgs
            if target == i:
                img_mnist = img_mnist.to(device)
                img_svhn = img_svhn.to(device)
                img_3 = img_3.to(device)
                img_4 = img_4.to(device)
                img_5 = img_5.to(device)
                # text = text.to(flags.device);
                # samples.append((img_mnist, img_svhn, text, target))
                samples.append((img_mnist, img_svhn, img_3, img_4, img_5, target))
                break
    outputs = []
    for mod in range(5):
        outputs_mod = [samples[digit][mod] for digit in range(10)]
        outputs.append(torch.stack(outputs_mod, dim=0))
    return outputs


class Flatten(torch.nn.Module):
    """
    Helper function for ClfImg class
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClfImg(nn.Module):
    """
    PolyMNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10)  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        return F.log_softmax(h, dim=-1)


# Helper function to plot captions so to smoothly log on WandB
def plot_text_as_image_tensor(sentences_lists_of_words, pixel_width=64, pixel_height=384):
    imgs = []
    for sentence in sentences_lists_of_words:
        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(pixel_width * px, pixel_height * px))
        plt.text(
            x=1,
            y=0.5,
            s='{}'.format(
                ' '.join(i + '\n' if (n + 1) % 1 == 0
                         else i for n, i in enumerate([word for word in sentence.split() if word != '<eos>']))),
            fontsize=7,
            verticalalignment='center_baseline',
            horizontalalignment='right'
        )
        plt.axis('off')

        # Draw the canvas and retrieve the image as a NumPy array
        fig.canvas.draw()
        image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255  # Normalize to [0, 1]
        imgs.append(image_tensor)
        # Clean up the figure
        plt.close(fig)
    return torch.stack(imgs, dim=0)


def save_and_log_latents(latents, name, basepath, step=0, wandb_run=None, save_kwargs={"normalize": True}, wandb_kwargs={}):
    batch_size = latents.shape[0]
    latents = latents.detach().cpu()
    latents = torch.transpose(latents, 0, 1).reshape(-1, *latents.shape[2:]).unsqueeze(1)
    latents = make_grid(latents, nrow=batch_size)

    save_and_log_processed(latents, name, basepath, step, wandb_run, save_kwargs=save_kwargs, wandb_kwargs=wandb_kwargs)


def save_and_log_img(img, name, basepath, display_data_f=None,  step=0, wandb_run=None, save_kwargs={}, wandb_kwargs={}, nrow=8):
    if isinstance(img, torch.distributions.distribution.Distribution):
        img = distribution_to_tensor(img).detach().cpu()
    if display_data_f is not None:
        img = torch.stack([display_data_f(d.detach().cpu()) for d in img], dim=0)

    img_grid = make_grid(img, nrow=nrow)

    save_and_log_processed(img_grid, name, basepath, step, wandb_run, save_kwargs=save_kwargs, wandb_kwargs=wandb_kwargs)


def save_and_log_processed(img, name, basepath, step=0, wandb_run=None, save_kwargs={}, wandb_kwargs={}):
    save_path = Path(basepath).joinpath(f"step_{step}_"+name+".png")
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    save_image(img, save_path, **save_kwargs)

    if wandb_run is not None:
        wandb_run.log({name: wandb.Image(img)}, step=step, **wandb_kwargs)


def distribution_to_tensor(dist):
    if isinstance(dist, torch.distributions.Bernoulli):
        tens = dist.mode
    else:
        tens = dist.mean
    return tens


def duplicate_stable_diff_weights(sd_weights, vae, alpha=0.25):
    # If we have single multimodal VAE, we double the channel dimensions and duplicate convolutional weights across the layers.
    # to allow the model to break symmetry we mix in the duplicated channels with the initialised weights from vae

    target_dict = vae.state_dict()
    for source_key, source_value in sd_weights.items():
        if source_key not in target_dict:
            # del sd_weights[source_key]
            # print(f"Deleting key: {source_key}")
            continue

        if target_dict[source_key].shape != source_value.shape:
            if (len(target_dict[source_key].shape) == len(source_value.shape) == 4):
                source_shape = source_value.shape
                # Conv weights
                out_mult = float(target_dict[source_key].shape[0]) / source_value.shape[0]
                in_mult = float(target_dict[source_key].shape[1]) / source_value.shape[1]
                # Pretty sure everthing goes wrong if out_mult and in_mult are not equal

                source_value = source_value.repeat(math.ceil(out_mult), math.ceil(in_mult), 1, 1)
                source_value = source_value[:target_dict[source_key].shape[0], :target_dict[source_key].shape[1], :, :]

                interpolation_const = torch.ones_like(source_value) * alpha
                interpolation_const[:source_shape[0], :source_shape[1], :, :] = 0.0

                source_value = interpolation_const * target_dict[source_key] + (1 - interpolation_const) * source_value
                source_value = source_value / math.sqrt(out_mult)
                print(f"Duplicated conv weights: {source_key} by input factor {in_mult} and output factor {out_mult}")

            elif (len(target_dict[source_key].shape) == len(source_value.shape) == 1):
                # Conv bias
                source_shape = source_value.shape
                out_mult = float(target_dict[source_key].shape[0]) / source_value.shape[0]

                source_value = source_value.repeat(math.ceil(out_mult))
                source_value = source_value[:target_dict[source_key].shape[0]]

                interpolation_const = torch.ones_like(source_value) * alpha
                interpolation_const[:source_shape[0]] = 0.0

                source_value = interpolation_const * target_dict[source_key] + (1 - interpolation_const) * source_value

                source_value = source_value / out_mult
                print(f"Duplicated bias: {source_key} by output factor {out_mult}")

            else:
                print(f"Could not duplicate weights for layer: {source_key}")

            sd_weights[source_key] = source_value

    return sd_weights


def insert_stable_diff_weights(sd_weights, vae):
    # If we have single multimodal VAE, we insert the stable diffusion weights into the channels but dont duplicate them since this never breaks symmetry

    target_dict = vae.state_dict()
    for source_key, source_value in sd_weights.items():
        if source_key not in target_dict:
            # del sd_weights[source_key]
            # print(f"Deleting key: {source_key}")
            continue

        if target_dict[source_key].shape != source_value.shape:
            if (len(target_dict[source_key].shape) == len(source_value.shape) == 4):

                target_dict[source_key][:source_value.shape[0], :source_value.shape[1], :, :] = source_value
                print(f"Inserted conv weights: {source_key} with shape {source_value.shape} and output factor {target_dict[source_key].shape}")

            elif (len(target_dict[source_key].shape) == len(source_value.shape) == 1):
                # Conv bias
                target_dict[source_key][:source_value.shape[0]] = source_value
                print(f"Inserted conv weights: {source_key} by with shape {source_value.shape} and output factor {target_dict[source_key].shape}")
            else:
                print(f"Could not duplicate weights for layer: {source_key}")
        else:
            target_dict[source_key] = source_value

    return target_dict


def expand_sub_confs(conf):
    # Expand sub-configs
    for key, val in conf.items():
        if key[-4:] == "conf":
            subconf = OmegaConf.load(val)
            conf[key] = expand_sub_confs(subconf)
    return conf


def move_data_to_device(data, device="cuda"):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) or isinstance(data, list):
        return [move_data_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    else:
        return data


def override_config(conf, overrides):
    for key, val in overrides.items():
        print(f"overriding : {key} with value {val}")
        key_stages = key.split(".")

        if any(isinstance(val, t) for t in [np.int64, np.str_, np.float32, np.float64]):
            val = val.item()

        try:
            conf_val = conf
            for k in key_stages[:-1]:
                if "[" in k:
                    # If there is an array index
                    conf_val = conf_val[k.split("[")[0]]
                    conf_val = conf_val[int(k.split("[")[1][:-1])]
                else:
                    conf_val = conf_val[k]

            if "[" in key_stages[-1]:
                # handle array index
                conf_val = conf_val[key_stages[-1].split("[")[0]]
                conf_val[int(key_stages[-1].split("[")[1][:-1])] = val
            else:
                conf_val[key_stages[-1]] = val

        except (IndexError, KeyError):
            print(f"Could not override key: {key}")

    return conf

def save_checkpoint(ema, model, optimiser, scheduler, iteration, path, **kwargs):
    sd = {"ema": ema.state_dict() if ema is not None else None,
          "model": model.state_dict() if model is not None else None,
          "optimiser": optimiser.state_dict() if optimiser is not None else None,
          "scheduler": scheduler.state_dict() if scheduler is not None else None,
          "iteration": iteration if iteration is not None else None,
          **{name: val.state_dict() for name, val in kwargs.items()}}

    torch.save(sd, path)


def load_checkpoint(ema, model, optimiser, scheduler, path, map_location='cpu', **kwargs):
    sd = torch.load(path, map_location=map_location)

    try:
        if "model" in sd:
            model.load_state_dict(sd["model"])
        else:
            logger.warning(f"No model key found in checkpoint, assuming only the model is saved")
            model.load_state_dict(sd)
    except RuntimeError:
        # one of the checkpoints is saved slightly differently for some reason
        if "model" in sd:
            model.module.load_state_dict(sd["model"])
        else:
            logger.warning(f"No model key found in checkpoint, assuming only the model is saved")
            model.module.load_state_dict(sd)

    if ("ema" in sd) and (ema is not None):
        ema.load_state_dict(sd["ema"])
    else:
        logger.warning("No EMA state dict found in checkpoint")

    if ("optimiser" in sd) and (optimiser is not None):
        optimiser.load_state_dict(sd["optimiser"])
    else:
        logger.warning("No optimiser state dict found in checkpoint")

    if ("scheduler" in sd) and (scheduler is not None):
        scheduler.load_state_dict(sd["scheduler"])
    else:
        logger.warning("No scheduler state dict found in checkpoint")

    for key, val in kwargs.items():
        if key in sd:
            if hasattr(sd[key], "state_dict"):
                # We accidentally saved the whole model!
                val.load_state_dict(sd[key].state_dict())
            else:
                val.load_state_dict(sd[key])

    return sd["iteration"] if "iteration" in sd else 0


def load_image_datasets(data_conf, precompute_latents=False):
    if isinstance(data_conf, ListConfig):
        # We have multiple datasets to concat
        train_datasets, test_datasets = list(zip(*[load_image_datasets(dc, precompute_latents) for dc in data_conf]))

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)

        return train_dataset, test_dataset

    print(data_conf)
    tx = {"colour": v2.Compose([v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)]),
          "semseg": v2.Compose([v2.Resize(data_conf.input_size, interpolation=InterpolationMode.NEAREST_EXACT),
                                v2.CenterCrop(data_conf.input_size)]),
          "depth": v2.Compose([v2.Resize(data_conf.input_size), v2.CenterCrop(data_conf.input_size)])}

    return_class = "return_class" in data_conf and data_conf.return_class

    train_opts = OmegaConf.to_container(data_conf)
    test_opts = copy.deepcopy(OmegaConf.to_container(data_conf))
    for key in ["p_hor_flip", "p_colour_jitter"]:
        test_opts.pop(key, None)
        if precompute_latents:
            # Dont augment if precomputing latents
            train_opts.pop(key, None)

    train_dataset = get_dataset(data_conf.dataset_name,
                                transform=tx,
                                split="train",
                                return_row=precompute_latents,
                                return_class=return_class,
                                **train_opts)
    test_dataset = get_dataset(data_conf.dataset_name,
                               transform=tx,
                               split="val",
                               return_row=precompute_latents,
                               return_class=return_class,
                               **test_opts)

    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size, world_size=1, rank=0, data_parallel=False):
    kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                       shuffle=True) if data_parallel else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank,
                                      shuffle=False) if data_parallel else None

    train_collate_fn = None # train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None 
    test_collate_fn = None # test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=not data_parallel, sampler=train_sampler,
                                               collate_fn=train_collate_fn, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              sampler=test_sampler,
                                              collate_fn=test_collate_fn, **kwargs)

    return train_loader, test_loader


def cat_dists_or_tensor_latents(latents):
    if all([isinstance(l, torch.Tensor) for l in latents]):
        # Tensorble
        latents = torch.cat(latents, dim=1)
    else:
        # Distributions
        latents = torch.cat([l.mean for l in latents], dim=1)
    return latents


class DiversityMetrics():

    def __init__(self, name):
        self.name = name
        if name == "tie":
            self.metric = InceptionMetrics()
        elif name == "tce":
            self.metric = ClipMetrics()

    def __call__(self, folder, **kwargs):
        if self.name == "tie":
            return self.metric.tie(folder)
        elif self.name == "tce":
            return self.metric.tce(folder)

