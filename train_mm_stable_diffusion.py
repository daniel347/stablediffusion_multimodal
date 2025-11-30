# Train MMVAEplus on CUB Image-Captions dataset
import os
import shutil
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
import tqdm
from collections.abc import Iterable
import copy
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use('Agg')
import wandb

import torch
from torch import optim
from torchvision.transforms import v2, InterpolationMode
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler, ExponentialLR, CosineAnnealingWarmRestarts, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.utils as tvu
from torch.amp import GradScaler, autocast
from torchvision.transforms import Resize, InterpolationMode

"""
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ray.cloudpickle as pickle
"""
import tempfile
import sys
import glob
import logging

from ldm.util import instantiate_from_config, Timer, GradAutoClipper, count_params
from ldm.data import get_dataset, MMCelebA
from objectives.objective_util import MMDGaussianLoss
from ldm.modules.ema import DummyEMA
import pyiqa
from pyiqa.archs.fid_arch import frechet_distance, kernel_distance
import cmmd_pytorch as cmmd
from utils import (save_and_log_latents, save_and_log_img, expand_sub_confs, move_data_to_device, override_config,
                   save_checkpoint, load_checkpoint, load_image_datasets, create_dataloaders,
                   cat_dists_or_tensor_latents,
                   DiversityMetrics, DisplayMAELatent)
from reconstruct_model_phema import reconstruct_phema
import ldm.models.diffusion as diffusion
import cv2
from mae.distribution import DiagonalGaussian

import random
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

import math

enable_wandb = True
optimise_hyperparameters = False
log_iterations = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def load_precomputed_datasets(conf_data, latent_path_train, latent_path_val):
    train_dataset = get_dataset("latents", latent_path=latent_path_train, split="train", **conf_data)
    test_dataset = get_dataset("latents", latent_path=latent_path_val, split="val", **conf_data)

    return train_dataset, test_dataset


def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr


class DiffusionIntermediateLogger():
    def __init__(self, save_freq, n_save=4):
        self.save_freq = save_freq
        self.n_save = n_save
        self.x0t_imgs = []
        self.ts = []

    def save_intermediate_z0t_callback(self, pred_x0, t):
        if t % self.save_freq == 0:
            self.x0t_imgs.append(pred_x0[:self.n_save].detach().cpu())
            self.ts.append(t)

    def save_and_log(self, path, step, run, clear=True):
        x0t_stack = torch.stack(self.x0t_imgs)
        for n in range(self.n_save):
            save_and_log_latents(x0t_stack[:, n], f"Generation/intermediate_z0t_img{n}", path, step, run)

        if clear:
            self.x0t_imgs = []
            self.ts = []


def initialise_model(conf):
    OmegaConf.to_container(conf["model"])
    model = instantiate_from_config(conf["model"])

    if conf.init_from_stable_diffusion:
        # sd_state_dict = torch.load("checkpoints/sd-v1-4.ckpt")
        model.init_from_ckpt("checkpoints/v2-1_512-ema-pruned.ckpt")

    return model


def generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode, sample=True, return_dist=False, cls_tokens=False):
    avg_z = z_mode == "avg"
    random_z = z_mode == "random"
    channel_u = (model_first_stage.w_channels * NUM_VAES + model_first_stage.z_channels) if model_first_stage.factorised_latent_space else model_first_stage.u_len
    model_opts = {"avg_z": avg_z, "random_z": random_z, "return_dist": sample, "total_mask_ratio": 0.0, "shared_mask_ratio": 0.0, "no_mask": True}
    split_dim = -1 if getattr(model_first_stage, "masked_autoencoder", False) else 1

    if conf.training.precompute_latents:
        latents = data
    else:
        latents = model_first_stage.encode(data, **model_opts)
        if getattr(model_first_stage, "masked_autoencoder", False):
            if cls_tokens:
                cls = latents[3]
            latents = latents[0] + latents[1]  # W + Z
   
    if isinstance(latents, dict) and "mean" in latents and "std" in latents:
        # Assume dict of mean and std of Normal distribution
        if sample:
            print("sampling latents")
            if cls_tokens:
                cls = torch.distributions.Normal(latents["cls_mean"], latents["cls_std"]).sample()
            latents = torch.distributions.Normal(latents["mean"], latents["std"]).sample()
            sample = False  # We have finished sampling, so don't try to sample again
        elif return_dist:
            if cls_tokens:
                cls = torch.distributions.Normal(latents["cls_mean"], latents["cls_std"])
                latents = torch.distributions.Normal(latents["mean"], latents["std"])
                return latents, cls
            latents = torch.distributions.Normal(latents["mean"], latents["std"])
            return torch.distributions.Normal(latents["mean"], latents["std"])
        else:
            latents = latents["mean"]
            if cls_tokens:
                cls = latents["cls_mean"]

    elif isinstance(latents, dict) and "latent" in latents:
        if cls_tokens:
            cls = latents["cls"]
        latents = latents["latent"]

    if isinstance(latents, torch.Tensor):
        if latents.shape[1] == channel_u:
            if cls_tokens:
                return latents, cls
            return latents
        else:
            latents = torch.split(latents, [model_first_stage.w_channels for _ in range(NUM_VAES)] + [model_first_stage.z_channels for _ in range(NUM_VAES)], dim=split_dim)
        
    if isinstance(latents, list) or isinstance(latents, tuple):
        if len(latents) > NUM_VAES + 1:
            # TODO: z_mo
            assert (random_z or isinstance(z_mode, int)), "One of average or random z must be set or a integer supplied to choose one z, or the z-combiner must be used"
            if random_z:
                latents = [*latents[:NUM_VAES], random.choice(latents[NUM_VAES:])]
            elif isinstance(z_mode, int):
                latents = [*latents[:NUM_VAES], latents[NUM_VAES + z_mode]]

        if sample:
            print("sampling chosen latents")
            latents = cat_dists_or_tensor_latents([l.sample() for l in latents], dim=split_dim)
        elif return_dist:
            print("returning chosen latents as distributions")
            assert not getattr(model_first_stage, "masked_autoencoder", False), "Cannot return dists for split latents with masked autoencoder yet"
            means = cat_dists_or_tensor_latents(latents, dim=split_dim)
            try:
                # DiagonalGaussianDistribution has std
                stds = cat_dists_or_tensor_latents([l.std for l in latents], dim=split_dim)
            except AttributeError:
                # torch.distributions.Normal does not have std, so we use stddev
                stds = cat_dists_or_tensor_latents([l.stddev for l in latents], dim=split_dim)
            return torch.distributions.Normal(means, stds)
        else:
            print("concatenating chosen latents")
            latents = cat_dists_or_tensor_latents(latents, dim=split_dim)

        if getattr(model_first_stage, "masked_autoencoder", False):
            # Reshape to an image shape (B, C, H, W)
            N, L, T = latents.shape
            side_len = int(math.sqrt(L))
            latents = latents.permute((0, 2, 1)).reshape((N, T, side_len, side_len))

        if cls_tokens:
            return latents, cls
        return latents


def run_training(rank, conf, data_parallel=False, world_size=None, ray_tune=False, enable_pbars=True):
    try:
        if data_parallel:
            setup(rank, world_size)

        print("Working directory: ", os.getcwd())

        use_mixed_precision = conf.training.use_mixed_precision
        conf["model"]["params"]["scorenet_config"]["params"]["use_fp16"] = use_mixed_precision
        if use_mixed_precision:
            print("Using mixed precision training")
            scaler = GradScaler()

        if conf.training.latent_scale_per_channel:
            conf.model.params.first_stage_config.params.scale_per_channel = True

        # Initialise the model
        if "ckpt_path" not in conf.model.params.first_stage_config.params or \
                conf.model.params.first_stage_config.params.ckpt_path is None:
            logger.warning("No checkpoint path for first stage! This is could be unintentional")
        model = initialise_model(conf)
        n_param = count_params(model)
        print(f"Full model has {n_param*1.e-6:.2f} M parameters")

        if conf.function.name == "count_params":
            sys.exit(0)

        device = rank
        model = model.to(device)

        if data_parallel:
            print(f"Initialising data-based parallelism with {len(conf.device)} GPUs")
            model = DDP(model, device_ids=[rank])

        if conf.training.t_sampler is not None:
            t_sampler_conf = OmegaConf.to_container(conf.training.t_sampler)
            t_sampler_conf["params"]["sigma_sched"] = torch.sqrt((1 - model.alphas_cumprod)/model.alphas_cumprod)
            t_sampler = instantiate_from_config(t_sampler_conf)
        else:
            t_sampler = None

        model_first_stage = model.module.first_stage_model if data_parallel else model.first_stage_model
        model_scorenet = model.module.model if data_parallel else model.model

        NUM_VAES = len(model_first_stage.vaes)
        print(f"Using {NUM_VAES} vaes")

        if "ema" in conf:
            logger.info(f"Using EMA {conf.ema.target}")
            ema_conf = OmegaConf.to_container(conf.ema)
            ema_conf["params"]["net"] = model
            ema = instantiate_from_config(ema_conf)
        else:
            ema = DummyEMA(model)

        if "sampler" in conf:
            sampler_conf = OmegaConf.to_container(conf.sampler)
            if "ddpm" in sampler_conf["target"]:
                sampler = model
            else:
                sampler_conf["params"]["model"] = model
                sampler_conf["params"]["device"] = device
                sampler = instantiate_from_config(sampler_conf)

        if "guidance" in conf:
            meas_models = []
            for meas_conf in conf.guidance.meas_confs:
                if meas_conf is None:
                    meas_models.append(None)
                else:
                    meas_models.append(instantiate_from_config(meas_conf))

            guider_conf = OmegaConf.to_container(conf.guidance.guider)
            guider_conf["params"]["model_first_stage"] = model_first_stage
            guider_conf["params"]["measurement_models"] = meas_models

            guider = instantiate_from_config(guider_conf)
        else:
            guider = None


        # Set experiment name if not set
        if not conf.experiment:
            conf.experiment = model.modelName

        # Set up run path
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        if conf.training.test_only:
            runId += "_test"

        if ray_tune:
            runId += f"_rank{train.get_context().get_world_rank()}"

        if conf.name is not None:
            runId += f"_{conf.name}"

        if rank == 0 or not data_parallel:
            experiment_dir = Path(os.path.join(conf.outputdir, conf.experiment, "checkpoints"))
            experiment_dir.mkdir(parents=True, exist_ok=True)
            runPath = experiment_dir / runId
            if runPath.exists():
                shutil.rmtree(runPath)
            runPath.mkdir(parents=True, exist_ok=True)
            print('Expt:', runPath)
            print('RunID:', runId)

            # save args to run
            with open('{}/args.json'.format(runPath), 'w') as fp:
                OmegaConf.save(config=conf, f=fp)

        # WandB
        if enable_wandb:
            wandb.login()
            run = wandb.init(
                # Set the project where this run will be logged
                project=conf.experiment,
                # Track hyperparameters and run metadata
                config=OmegaConf.to_container(conf),
                # Run name
                name=f"{runId}_{str(rank)}",
                group=f"DDP_{runId}",
                notes=conf.notes,
            )
            wandb.alert(title=f"Experiment {conf.experiment} has started", text=f"Run id: {runId}")
        else:
            run = None

        # preparation for training
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=conf.training.lr, amsgrad=True)

        grad_clipper = None
        if "grad_clipper" in conf.training:
            grad_clipper = GradAutoClipper(model, **conf.training.grad_clipper)


        scheduler = ChainedScheduler([LinearLR(optimizer, 0.1, 1, total_iters=200),
                                      CosineAnnealingWarmRestarts(optimizer, 4000, 2, eta_min=0.1 * conf.training.lr)])
                                      # ExponentialLR(optimizer, conf.training.gamma)])

        scheduler_step_freq = 1

        start_iter = None

        if conf.model_initialisation is not None:
            start_iter = load_checkpoint(ema, model, optimizer, scheduler, conf.model_initialisation)

        if ray_tune:
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    start_iter = load_checkpoint(ema, model, optimizer, scheduler, data_path)

        if start_iter is None:
            start_iter = conf.training.start_iter

        test_param_name = "model.input_blocks.4.0.in_layers.0.weight"

        train_dataset, test_dataset = load_image_datasets(conf.data, conf.training.precompute_latents)

        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, conf.training.batch_size, world_size, rank, data_parallel)

        data_conf_0 = conf.data[0] if isinstance(conf.data, ListConfig) else conf.data
        data_root_dirs = [Path(cd.root_data_dir) for cd in conf.data] if isinstance(conf.data, ListConfig) else [Path(conf.data.root_data_dir)]
        dataset_name = conf.dataset_name # "_".join([cd.dataset_name for cd in conf.data]) if isinstance(conf.data, ListConfig) else conf.data.dataset_name
        
        if isinstance(train_dataset, torch.utils.data.ConcatDataset):
            display_data_fs = [getattr(train_dataset.datasets[0], f"{mode}_display") for mode in data_conf_0.modality]
        else:
            display_data_fs = [getattr(train_dataset, f"{mode}_display") for mode in data_conf_0.modality]

        if start_iter > 0 and start_iter < 50:
            # this is almost certainly an epoch number
            print("converting start epoch to start iter")
            start_iter *= len(train_loader)

        already_precomputed = False
        if conf.training.precompute_latents:
            if conf.function.name != "training_loop":
                print(" !!!!! Warning : Precompute latents is set but function is not training loop - this can cause problems !!!!! ")

            latents_train_path = Path(conf.training.precompute_latents_path + "_train") / conf.dataset_name
            latents_val_path = Path(conf.training.precompute_latents_path + "_val") / conf.dataset_name
            if latents_train_path.exists() and latents_val_path.exists() and not conf.training.force_recompute:
                # We have already computed the latents

                train_dataset, test_dataset = load_precomputed_datasets(data_conf_0, latents_train_path, latents_val_path)
                train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, conf.training.batch_size, world_size, rank,
                                                               data_parallel)
                print(f"Latent dataset length train: {len(train_dataset)} val: {len(test_dataset)}")
                already_precomputed = True

        return_class = "return_class" in data_conf_0 and data_conf_0.return_class
        return_index = conf.training.precompute_latents
        modality = data_conf_0.modality
        avg_z = conf.training.z_mode == "avg"
        random_z = conf.training.z_mode == "random"
        z_combiner = conf.training.z_mode == "z_combiner"
        sample = (not getattr(model_first_stage, "masked_autoencoder", False)) or getattr(model_first_stage, "is_variational", False)
        
        if z_combiner:
            model_first_stage.use_z_comb_network = True

        dataloaders = {"train_dataset": train_dataset,
                       "test_dataset": test_dataset,
                       "train_loader": train_loader,
                       "test_loader": test_loader}

        if getattr(model_first_stage, "masked_autoencoder", False):
            mae_latent_disp = DisplayMAELatent()

        n_cls_tokens = getattr(model_scorenet, "n_cls_tokens", 0)

        def fit_mae_latent_disp(dataloaders, split="train", n_batches=20):
            data_loader = dataloaders["train_loader" if split == "train" else "test_loader"]
            all_latents = []

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader),
                                 position=rank if data_parallel else 0)
            else:
                pbar = enumerate(data_loader)

            with (torch.no_grad()):
                for i, dataT in pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    latents = generate_latents(conf, model_first_stage, data, NUM_VAES,
                                               z_mode=conf.training.z_mode,
                                               sample=sample,
                                               cls_tokens=False)
                    latents = model_first_stage.scale_latents(latents)
                    all_latents.append(latents)

                    if i >= n_batches:
                        break

            latents = torch.cat(all_latents, dim=0)
            mae_latent_disp.fit(latents)


        def get_latent_scale_factor(dataloaders, split="train", n_batches=1, per_channel=True):

            data_loader = dataloaders["train_loader" if split == "train" else "test_loader"]

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader),
                                 position=rank if data_parallel else 0)
            else:
                pbar = enumerate(data_loader)

            channel_means = 0
            channel_vars = 0

            with (torch.no_grad()):
                for i, dataT in pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    latents = generate_latents(conf, model_first_stage, data, NUM_VAES,
                                               z_mode=conf.training.z_mode,
                                               sample=sample,
                                               cls_tokens=False)

                    avg_dims = [0, 2, 3]
                    if not per_channel:
                        avg_dims += [1]

                    channel_means += latents.mean(dim=avg_dims)
                    channel_vars += latents.var(dim=avg_dims)

                    if i >= n_batches:
                        break

            channel_means = channel_means / n_batches
            channel_stds = torch.sqrt(channel_vars / n_batches)

            return channel_means, channel_stds

        def precompute_latents(dataloaders, split="train", save_folder="latents", dataset_name=None):
            dataloader = dataloaders["train_loader" if split == "train" else "test_loader"]
            model.eval()

            if data_parallel:
                dataloader.sampler.set_epoch(0)

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                                 position=rank if data_parallel else 0)
            else:
                pbar = enumerate(dataloader)

            out_base_folder = Path(f"{save_folder}_{split}")
            if dataset_name is not None:
                out_base_folder = out_base_folder / dataset_name

            with torch.no_grad():
                for i, dataT in pbar:
                    # print(len(dataT))
                    # print(len(dataT[0]))
                    if return_class:
                        data, classes, row = move_data_to_device(dataT, device=device)
                    else:
                        data, row = move_data_to_device(dataT, device=device)
                    # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                    optimizer.zero_grad()

                    # TODO: A slightly gross mix of MAE and VAE options
                    model_opts = {"avg_z": avg_z, "total_mask_ratio": 0.0, "shared_mask_ratio": 0.0, "return_dist": True, "no_mask": True}

                    if use_mixed_precision:
                        with autocast(dtype=torch.float16, device_type="cuda"):
                            latents = model_first_stage.encode(data, **model_opts)
                    else:
                        latents = model_first_stage.encode(data, **model_opts)

                    dists = False
                    if getattr(model_first_stage, "masked_autoencoder", False):
                        if getattr(model_first_stage, "is_variational", False):
                            dists = True
                            dist_dim = -1
                            cls_tokens = DiagonalGaussian.cat_dists(latents[3], dim=1) if n_cls_tokens > 0 else None
                            cls_tokens = torch.cat([cls_tokens.mean, cls_tokens.std], dim=-1)

                            latents = DiagonalGaussian.cat_dists(latents[0] + latents[1], dim=-1)
                            latents = torch.cat([latents.mean, latents.std], dim=-1)  # W + Z
                        else:
                            cls_tokens = torch.cat(latents[3], dim=1) if n_cls_tokens > 0 else None
                            latents = torch.cat(latents[0] + latents[1], dim=-1)  # W + Z
                    else:
                        if isinstance(latents, list):
                            if isinstance(latents[0], torch.Tensor):
                                latents = torch.cat(latents, dim=1)
                                print(latents.shape)
                            else:
                                # We have distributions
                                print("Latents are distributions")
                                dists = True
                                dist_dim = 0
                                latents = torch.cat([l.mean for l in latents] + [l.std for l in latents], dim=1)
                                # latents = {"mean": latent_means, "std": latent_stds}  # Means then stds
                                # print(f"latent meat shape: {latent_means.shape} std shape: {latent_stds.shape}")
                                print(latents.shape)

                    for k, (latent, r) in enumerate(zip(latents, row[f"{modality[0]}_path"])):
                        save_path = Path(r.replace(modality[0], "latent"))
                        for root in data_root_dirs:
                            if save_path.is_relative_to(root):
                                save_path = out_base_folder / save_path.relative_to(root.parent)  # keep the dataset folder name
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        # print(save_path)
                        data_to_save = {}

                        if dists:
                            mean, std = torch.chunk(latent, 2, dim=dist_dim)
                            data_to_save = {"mean": mean.contiguous().cpu(), "std": std.contiguous().cpu()}
                            if n_cls_tokens > 0:
                                cls_mean, cls_std = torch.chunk(cls_tokens[k], 2, dim=-1)
                                data_to_save = {"cls_mean": cls_mean.contiguous().cpu(),
                                                "cls_std": cls_std.contiguous().cpu(),
                                                **data_to_save}
                        else:
                            data_to_save = {"latent": latent.contiguous().cpu()}
                            if n_cls_tokens > 0:
                                data_to_save["cls"] = cls_tokens[k].contiguous().cpu()

                        if return_class:
                            data_to_save["class"] = classes[r].contiguous().cpu()

                        torch.save(data_to_save, save_path.with_suffix(".pt"))


        def train_epoch(dataloaders, subepoch, itercount, epochcount, n_iter, train_pbar):
            model.train()
            b_loss = 0
            n_stopiter = n_iter + itercount

            while itercount < n_stopiter:
                if data_parallel:
                    dataloaders["train_loader"].sampler.set_epoch(epochcount)

                if enable_wandb:
                    wandb.log({"epoch": epochcount}, step=itercount)

                for i, dataT in train_pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)
                    # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                    optimizer.zero_grad()

                    latents = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode,
                                               sample=sample,
                                               cls_tokens=n_cls_tokens)

                    forward_kwargs = {}
                    if n_cls_tokens > 0:
                        latents, cls_tokens = latents
                        forward_kwargs["cls"] = cls_tokens

                    latents = model_first_stage.scale_latents(latents)

                    # print(f"Latent scale: {latents.std(dim=(0, 2, 3))}   mean: {latents.mean(dim=(0, 2, 3))}")

                    if use_mixed_precision:
                        with autocast(dtype=torch.float16, device_type="cuda"):
                            loss, loss_dict = model(latents, c=(classes if return_class else None), t_sampler=t_sampler, **forward_kwargs)

                        scaler.scale(loss).backward()

                        if grad_clipper is not None:
                            scaler.unscale_(optimizer)
                            grad_clipper.autoclip_grad()
                        scaler.step(optimizer)
                    else:
                        loss, loss_dict = model(latents, c=(classes if return_class else None), t_sampler=t_sampler, **forward_kwargs)
                        loss.backward()

                        if grad_clipper is not None:
                            grad_clipper.autoclip_grad()
                        optimizer.step()

                    if torch.isnan(loss).item():
                        if enable_wandb:
                            wandb.alert(title="Loss is NaN", text=f"Halting training at epoch {epochcount} and step {i}")
                        raise ValueError("Loss has become NaN, halting training")

                    b_loss += loss.detach().item()
                    if conf.print_freq > 0 and i % conf.print_freq == 0 and enable_pbars:
                        train_pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))

                    if ema is not None:
                        ema.update(conf.training.batch_size)

                    """
                    for a in range(len(ema.emas)):
                        print("Checking emas")
                        if not torch.all(ema.emas[a][test_param_name].data != model.state_dict()[test_param_name].data):
                            print(f"EMA {a} matches model parameters in iteration {i}.")
                        for b in range(a):
                            if not torch.all(ema.emas[b][test_param_name].data != ema.emas[a][test_param_name].data):
                                print(f"EMA {b} matches EMA {a} parameters in iteration {i}.")
                    """

                    if i % scheduler_step_freq == 0:
                        scheduler.step()
                        # print(f"Learning rate: {scheduler.get_last_lr()}")

                    if enable_wandb and log_iterations:
                        wandb.log({"Loss/train": loss, "lr": scheduler.get_last_lr()[0]}, step=itercount)

                    if i == 0:
                        # test decode the saved latents
                        if getattr(model_first_stage, "masked_autoencoder", False):
                            latents_unscaled = model_first_stage.scale_latents(latents, reverse=True)
                            decoded_imgs = model_first_stage.decode_no_mask(latents_unscaled, cls_tokens=cls_tokens)[0]

                            for m, (name, dbatch) in enumerate(zip(modality, decoded_imgs)):
                                save_and_log_img(dbatch, f"DataDecoded/{name}", runPath,
                                                 display_data_fs[m], step=itercount, wandb_run=run)

                        if getattr(model_first_stage, "masked_autoencoder", False):
                            latents_disp = mae_latent_disp.transform(latents)
                            save_and_log_latents(latents_disp, f'Latents/u', runPath, itercount, run)
                        else:
                            # TODO: Find a way to visualise the latent space of the masked autoencoder - maybe a PCA idk?
                            if model_first_stage.factorised_latent_space:
                                *latents_w, latents_z = torch.split(latents, [model_first_stage.w_channels for _ in range(NUM_VAES)] + [model_first_stage.z_channels], dim=1)

                                for j, latent in enumerate(latents_w):
                                    save_and_log_latents(latent, f'Latents/wm{j}', runPath, itercount, run)
                                if latents_z.shape[1] > 0:
                                    save_and_log_latents(latents_z, f'Latents/z', runPath, itercount, run)
                            else:
                                save_and_log_latents(latents, f'Latents/u', runPath, itercount, run)
                        """
                        for m, (name, dbatch) in enumerate(zip(modality, latent)):
                            save_and_log_img(dbatch, f"Data/{name}", runPath,
                                             display_data_fs[m], step=itercount, wandb_run=run)
                        """

                    itercount += 1
                    if itercount >= n_stopiter:
                        break

                if itercount < n_stopiter:
                    print("Finished one epoch")
                    train_pbar = tqdm.tqdm(enumerate(dataloaders["train_loader"]), total=len(dataloaders["train_loader"]),
                                     position=rank if data_parallel else 0)
                    epochcount += 1

            # Epoch loss
            subepoch_loss = b_loss / n_iter
            if enable_wandb and not log_iterations: wandb.log({"Loss/train": subepoch_loss}, step=itercount)
            print('====> Subepoch: {:03d} Train loss: {:.4f}'.format(subepoch, subepoch_loss))

            return epochcount, train_pbar

        def test_epoch(dataloaders, subepoch, itercount, epochcount, reencode_end=False, save_intermediate_z0t=False, gen=True, t_sampler=None):
            if "sampler" in conf:
                run_params = conf.sampler.run_params
            else:
                run_params = {"S": 1000}  # For all but DDPM we must specify a number of timesteps, so we default to 1000

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(dataloaders["test_loader"]),
                                 total=len(dataloaders["test_loader"]),
                                 position=rank if data_parallel else 0)
            else:
                pbar = enumerate(dataloaders["test_loader"])

            if enable_wandb:
                wandb.log({"epoch": epochcount, "subepoch": subepoch}, step=itercount)

            b_loss = 0
            with (torch.no_grad(), ema.ema_scope("Epoch test")):
                for i, dataT in pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    latents = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode,
                                               sample=sample,
                                               cls_tokens=n_cls_tokens)

                    forward_kwargs = {}
                    if n_cls_tokens > 0:
                        latents, cls_tokens = latents
                        forward_kwargs["cls"] = cls_tokens

                    latents = model_first_stage.scale_latents(latents)

                    loss, loss_dict = model(latents, c=(classes if return_class else None), t_sampler=t_sampler, **forward_kwargs)
                    b_loss += loss.detach().item()

                    if i == 0 and gen:

                        # Perform test generations
                        if save_intermediate_z0t:
                            saver = DiffusionIntermediateLogger(10, n_save=16)
                            img_callback = saver.save_intermediate_z0t_callback
                        else:
                            img_callback = lambda x, t: None

                        if reencode_end:
                            def manifold_proj_f(z, t):
                                if (t[0].item()-1)//5 % 5 == 0 and t[0].item() > 400 and t[0].item() < 900:
                                    print("reencoding")
                                    z_scaled = model_first_stage.scale_latents(z, reverse=True)
                                    decs = [sx.mean for sx in model_first_stage.decode(z_scaled)]
                                    encs = model_first_stage.encode(decs)
                                    z_out = cat_dists_or_tensor_latents(encs)
                                    z_out = model_first_stage.scale_latents(z_out)
                                    return z_out
                                else:
                                    return z
                        else:
                            manifold_proj_f = lambda z, t: z

                        N_gen = 16
                        if return_class:
                            classes = torch.randint(low=0,
                                                    high=conf.model.params.scorenet_config.params.num_classes,
                                                    size=(N_gen,),
                                                    device=device)

                        samples_zw = sampler.sample(verbose=enable_pbars,
                                                    batch_size=N_gen,
                                                    shape=model.module.input_shape if data_parallel else model.input_shape,
                                                    conditioning=(classes if return_class else None),
                                                    img_callback=img_callback,
                                                    manifold_proj_f=manifold_proj_f,
                                                    n_cls=n_cls_tokens,
                                                    **run_params)
                        if n_cls_tokens > 0:
                            samples_zw, cls_tokens = samples_zw
                        else:
                            cls_tokens = None

                        # print(f"Generated latent scale: {samples_zw.std(dim=(0, 2, 3))}   mean: {samples_zw.mean(dim=(0, 2, 3))}")

                        samples_zw_scaled = model_first_stage.scale_latents(samples_zw, reverse=True)

                        if getattr(model_first_stage, "masked_autoencoder", False):
                            samples_x = model_first_stage.decode_no_mask(samples_zw_scaled, cls_tokens=cls_tokens)[0]
                        else:
                            samples_x = model_first_stage.decode(samples_zw_scaled)

                        if return_class:
                            wandb_kwargs = {"caption": "\n".join(
                                [f"{b}: {test_dataset.class_num_to_name(c)}" for b, c in enumerate(classes)])}
                        else:
                            wandb_kwargs = {}

                        if reencode_end:
                            samples_zw_2 = model_first_stage.encode([sx.mean for sx in samples_x], avg_z=avg_z)
                            samples_zw_2 = cat_dists_or_tensor_latents(samples_zw_2)
                            samples_x_2 = model_first_stage.decode(samples_zw_2)

                            for m in range(len(samples_x)):
                                save_and_log_img(samples_x_2[m], f'Generation/reencoded_m{m}',
                                                 runPath, display_data_fs[m], itercount, run, wandb_kwargs=wandb_kwargs,
                                                 save_kwargs={}, nrow=4)
                            save_and_log_latents(samples_zw_2, f'Generation/reencoded_latent',
                                                 runPath, itercount, run, wandb_kwargs=wandb_kwargs)

                        for m in range(len(samples_x)):
                            save_and_log_img(samples_x[m], f'Generation/m{m}',
                                             runPath, display_data_fs[m], itercount, run, wandb_kwargs=wandb_kwargs, save_kwargs={}, nrow=4)

                        save_and_log_latents(samples_zw, f'Generation/latent',
                                         runPath, itercount, run, wandb_kwargs=wandb_kwargs)

                        if save_intermediate_z0t:
                            saver.save_and_log(runPath, itercount, run)

            # Epoch test loss
            epoch_loss = b_loss / len(dataloaders["test_loader"].dataset)
            if enable_wandb: wandb.log({"Loss/test": epoch_loss}, step=itercount)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))

            return epoch_loss


        def test_loss_by_timestep(dataloaders):
            ts = torch.linspace(50, 950, 19, device=device).long()  # Test at these timesteps
            test_losses = []
            for i, t in enumerate(ts):
                t_sampler = lambda x: t.unsqueeze(0).expand(x.shape[0]).to(device)
                loss = test_epoch(dataloaders, 0, i, 0, gen=False, t_sampler=t_sampler)
                test_losses.append(loss)

                if enable_wandb:
                    wandb.log({"Loss/test": loss, "t": t.detach().cpu()}, step=i)

            test_losses = torch.tensor(test_losses)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ts.cpu().numpy(), test_losses.cpu().numpy())
            ax.set_xlabel("Timestep")
            ax.set.e632858_ylabel("Test Loss")

            fig.savefig(runPath / "timestep_wise_loss.png")


        def generate(dataloaders, subepoch, itercount, epochcount, n_samples=64, log_to_wandb=False, folder_name=None, save_latents=False):
            print(f"Using a latent mean of {model_first_stage.latent_mean} and latent scale of {model_first_stage.latent_scale}")

            if "sampler" in conf:
                run_params = conf.sampler.run_params
            else:
                run_params = {"S": 1000}  # For all but DDPM we must specify a number of timesteps

            gen_batch_size = 64
            n_batches = n_samples // gen_batch_size

            tempdirs = {}
            if folder_name is None:
                folder_name = f'temp_fid'
            for mod in modality:
                tempdir = os.path.join(runPath, f'{folder_name}_{mod}')
                os.makedirs(tempdir, exist_ok=True)
                tempdirs[mod] = tempdir

            if save_latents:
                tempdir = os.path.join(runPath, f'{folder_name}_latent')
                os.makedirs(tempdir, exist_ok=True)
                tempdirs["latent"] = tempdir

            if enable_pbars:
                pbar = tqdm.tqdm(range(n_batches))
            else:
                pbar = range(n_batches)

            model.eval()
            with (torch.no_grad(), ema.ema_scope("Epoch test")):
                for n in pbar:
                    if return_class:
                        classes = torch.randint(low=0,
                                                high=conf.model.params.scorenet_config.params.num_classes,
                                                size=(gen_batch_size,),
                                                device=device)

                    latents = sampler.sample(verbose=enable_pbars,
                                             batch_size=gen_batch_size,
                                             shape=model.module.input_shape if data_parallel else model.input_shape,
                                             conditioning=(classes if return_class else None),
                                             n_cls=n_cls_tokens,
                                             **run_params)

                    if n_cls_tokens > 0:
                        latents, cls_tokens = latents
                    else:
                        cls_tokens = None

                    latents = model_first_stage.scale_latents(latents, reverse=True)
                    if save_latents:
                        for b, latent in enumerate(latents):
                            save_path = Path(tempdirs["latent"]) / f"latent_{n * gen_batch_size + b}.pt"
                            data = {"latent": latent.contiguous().cpu()}
                            if n_cls_tokens > 0:
                                data["cls"] = cls_tokens.contiguous().cpu()
                            torch.save(data, save_path)

                    if getattr(model_first_stage, "masked_autoencoder", False):
                        samples_x = model_first_stage.decode_no_mask(latents, cls_tokens=cls_tokens)[0]
                    else:
                        samples_x = model_first_stage.decode(latents)

                    for m, (name, samples) in enumerate(zip(modality, samples_x)):
                        if isinstance(samples, torch.distributions.Distribution):
                            samples = samples.mean
                        for i, s in enumerate(samples):
                            s_disp = display_data_fs[m](s.detach().cpu(), opencv=True)
                            cv2.imwrite(os.path.join(tempdirs[name], f"sample_{name}_{n * gen_batch_size + i}.png"),
                                        s_disp)

                            if log_to_wandb:
                                step_num = ((n * gen_batch_size + i + 1) * len(dataloaders["train_loader"])) if log_iterations else n * gen_batch_size + i
                                run.log({f"Generation/m{m}": wandb.Image(s_disp)}, step=step_num)
            return tempdirs

        def test_metrics(dataloaders, subepoch, itercount, epochcount, metrics=["fid", "kid", "cmmd", "tce", "tie"], folder_name=None, gen_samples=True, **generate_kwargs):
            # print(f"Using a latent mean of {model_first_stage.latent_mean} and latent scale of {model_first_stage.latent_scale}")
            # raise ValueError

            if gen_samples:
                tempdirs = generate(dataloaders, subepoch, itercount, epochcount, folder_name=folder_name, **generate_kwargs)
            else:
                folder_path = Path(folder_name) if folder_name else Path(runPath) / 'temp_fid'
                tempdirs = {mod: f"{str(folder_path)}_{mod}" for mod in modality}
                
            metric_scores = {}
            for met in metrics:
                try:
                    if met == "cmmd":
                        metric = partial(cmmd.compute_cmmd, device=device)
                    elif met in ["tce", "tie"]:
                        metric = DiversityMetrics(met)
                    else:
                        metric = pyiqa.create_metric(met, device=device)
                except Exception as e:
                    print(e)
                    print(f"Error creating metric {met}, skipping")
                    continue

                scores = []
                for mod in modality:
                    if mod in ["colour", "depth"]:
                        split = "test" if dataset_name == "nyu_depth" else "val"
                        try:
                            score = metric(tempdirs[mod], dataset_name=f"{dataset_name}_{mod}", dataset_res=299, dataset_split=split, center_crop=True)
                            if enable_wandb:
                                wandb.log({f"{met}_{mod}": score, "epoch": epochcount, "subepoch": subepoch}, step=itercount)

                            scores.append(score)
                            print(f'====> {met} {mod}: {score:.2f}')
                        except Exception as e:
                            print(e)
                            print(f"Error computing {met} for {mod} in {dataset_name}, skipping")
                            scores.append(None)
                    else:
                        scores.append(None)

                metric_scores[met] = scores

            return metric_scores

        def test_optimum_EMA(dataloaders, out_nimgs, out_stds, compute_ema_models=True, **test_metric_kwargs):
            saves_path = os.path.dirname(conf.model_initialisation)
            print("saves_path", saves_path)

            out_nimgs, out_stds = zip(*product(out_nimgs, out_stds))

            model_names = [f"model_it{out_es//conf.training.batch_size}_ema_{out_std:.3f}.pt" for out_es, out_std in zip(out_nimgs, out_stds)]

            if compute_ema_models:
                it = reconstruct_phema(saves_path, out_std=out_stds, out_nimg=out_nimgs, out_dir=saves_path, epoch_size=conf.training.batch_size)
                for _ in it:
                    pass

            # latent_mean, latent_scale = get_latent_scale_factor(dataloaders, n_batches=100, per_channel=conf.training.latent_scale_per_channel)
            # model_first_stage.latent_mean = torch.nn.Parameter(latent_mean)
            # model_first_stage.latent_scale = torch.nn.Parameter(latent_scale)

            for step, name in enumerate(model_names):
                logging.info(f"Testing configuration with epochs:{out_nimgs[step]//conf.training.batch_size} and std:{out_stds[step]}")
                load_checkpoint(ema, model, optimizer, scheduler, os.path.join(saves_path, name))
                test_epoch(dataloaders, step, step, step)
                test_metrics(dataloaders, step, step, step, **test_metric_kwargs)

                if enable_wandb:
                    wandb.log({"n_epochs": out_nimgs[step], "ema_std": out_stds[step]}, step=step)

        def plot_timestep_logvars():
            logvars = model.logvar.cpu().detach().numpy()
            timesteps = np.arange(0, len(logvars))
            plot = sns.lineplot(x=timesteps, y=logvars)
            plot.set(xlabel="Diffusion timestep", ylabel="timestep log weighting")
            fig = plot.get_figure()
            fig.savefig(os.path.join(runPath, "timestep_logvars.png"))


        def test_compare_bisenet(dataloaders, step, n_samples=32):
            test_metrics(dataloaders, step, step, step, n_samples=n_samples, log_to_wandb=True)  # Generate images with our method
            colour_input_folder = os.path.join(runPath, f'temp_fid_colour')

            run_bisenet(step, colour_input_folder)


        def run_bisenet(epoch, img_dir=None, dataloader=None, out_folder_name="bisenet_output", n_imgs=32):
            bisenet_output_folder = os.path.join(runPath, out_folder_name)
            if not os.path.exists(bisenet_output_folder):
                os.mkdir(bisenet_output_folder)

            from BiSeNetFace.model import BiSeNet
            from BiSeNetFace.remap_classes import bisenet_to_mm_class_inds
            from PIL import Image

            out_folder_name = os.path.join(runPath, out_folder_name)
            if not os.path.exists(out_folder_name):
                os.mkdir(out_folder_name)

            cp = '79999_iter.pth'

            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.to(device)
            save_pth = os.path.join('BiSeNetFace/res/cp', cp)
            net.load_state_dict(torch.load(save_pth))
            net.eval()

            to_tensor = v2.Compose([
                v2.ToTensor(),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            B = 1 if img_dir is not None else conf.training.batch_size

            iter = dataloader if dataloader is not None else os.listdir(img_dir)
            if enable_pbars:
                iter = tqdm.tqdm(iter)

            with torch.no_grad():
                for i, data in enumerate(iter):
                    if isinstance(data, str):  # we have a file path
                        img = Image.open(os.path.join(img_dir, data))
                        orig_size = img.size

                        image = img.resize((512, 512), Image.BILINEAR)
                        image = to_tensor(image)
                        image = torch.unsqueeze(image, 0)
                        image = image.to(device)
                    else:
                        img = data[0]
                        orig_size = img.shape[2:]
                        image = v2.functional.resize(img, (512, 512), Image.BILINEAR, antialias=True)
                        image = v2.functional.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        image = image.to(device)


                    out = net(image)[0]
                    parsing = out.argmax(1)
                    s_disp = []
                    for p in parsing:
                        p = bisenet_to_mm_class_inds(p)
                        p = MMCelebA.index_semseg_to_one_hot(p)
                        s_disp.append(display_data_fs[1](p.detach().cpu()))

                    s_disp = torch.stack(s_disp, dim=0)
                    s_disp = v2.functional.resize(s_disp, orig_size, v2.InterpolationMode.BILINEAR, antialias=True)

                    for j, s in enumerate(s_disp):
                        img_path = os.path.join(out_folder_name, f"bisenet_{i*B + j}.png")
                        tvu.save_image(s, img_path)

                        if enable_wandb:
                            step_num = ((i*B + j + 1) * len(train_loader)) if log_iterations else i*B + j
                            run.log({"Generation/bisenet_m1": wandb.Image(s_disp)}, step=step_num)

                    if i*B >= n_imgs:
                        break

        def generate_conditional(dataloaders, n_samples=32, run_bisenet_on_input=False):
            if "sampler" in conf:
                run_params = conf.sampler.run_params
            else:
                run_params = {
                    "S": 1000}  # For all but DDPM we must specify a number of timesteps, so we default to 1000

            gen_batch_size = conf.training.batch_size  # Set this to control gen batch size

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(dataloaders["test_loader"]), total=len(dataloaders["test_loader"]), position=rank if data_parallel else 0)
            else:
                pbar = enumerate(dataloaders["test_loader"])

            model.eval()
            with ema.ema_scope("Epoch test"):
                for i, dataT in pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    y = []
                    for d, meas_model in zip(data, meas_models):
                        if meas_model is None:
                            y.append(None)
                        else:
                            y.append(meas_model(d))

                    latents = sampler.sample(verbose=enable_pbars,
                                             batch_size=gen_batch_size,
                                             shape=model.module.input_shape if data_parallel else model.input_shape,
                                             conditioning=None,
                                             guidance=guider,
                                             y=y,
                                             **run_params)

                    latents = model_first_stage.scale_latents(latents, reverse=True)
                    samples_x = model_first_stage.decode(latents)

                    print(f"latent shape: {latents.shape}")

                    save_and_log_latents(latents, f'Generation/latent_{i*gen_batch_size}', runPath, 0,
                                         run)

                    for m, (name, samples) in enumerate(zip(modality, samples_x)):
                        print(f"sample modality {m} shape: {samples.mean.shape}")
                        for k, s in enumerate(samples.mean):
                            save_and_log_img(s.unsqueeze(0), f'Generation/m{m}_{i*gen_batch_size + k}', runPath, display_data_fs[m], 0, run, save_kwargs={})

                    for m, (name, samples) in enumerate(zip(modality, data)):
                        for k, s in enumerate(samples):
                            save_and_log_img(s.unsqueeze(0), f'GT/m{m}_{i*gen_batch_size + k}', runPath, display_data_fs[m], 0, run, save_kwargs={})

                    for m, (name, samples) in enumerate(zip(modality, y)):
                        if samples is not None:
                            for k, s in enumerate(samples):
                                save_and_log_img(s.unsqueeze(0), f'y/m{m}_{i*gen_batch_size + k}', runPath, display_data_fs[m], 0, run, save_kwargs={})

                    if i * gen_batch_size >= n_samples:
                        break

            if run_bisenet_on_input:
                run_bisenet(0, dataloader=dataloaders["test_loader"])


        def check_latent_quality(dataloaders, n_latents=1024):
            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(dataloaders["train_loader"]),
                                 total=len(dataloaders["train_loader"]),
                                 position=rank if data_parallel else 0)
            else:
                pbar = enumerate(dataloaders["test_loader"])

            b_loss = 0
            all_latents = []
            all_latent_dists = []

            if conf.training.precompute_latents and not already_precomputed:
                conf.training.precompute_latents = False  # we want to compute latents on the fly

            if conf.training.compute_latent_scale:
                latent_mean, latent_scale = get_latent_scale_factor(dataloaders, n_batches=100,
                                                                    per_channel=conf.training.latent_scale_per_channel)
                model_first_stage.latent_mean = torch.nn.Parameter(latent_mean)
                model_first_stage.latent_scale = torch.nn.Parameter(latent_scale)
                print(f"found latent mean of: {latent_mean} and std of {latent_scale}")
                print("resaving the first stage model with latent mean and scale")
                # torch.save(model_first_stage.state_dict(),
                #            conf.model.params.first_stage_config.params.ckpt_path)

            with (torch.no_grad(), ema.ema_scope("Epoch test")):
                for i, dataT in pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    latent_dist = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode, return_dist=True, sample=False, cls_tokens=False)
                    latents = latent_dist.sample()
                    
                    print(f"latent dist means: {latent_dist.mean.mean(dim=(0, 2, 3))} stds: {latent_dist.stddev.mean(dim=(0, 2, 3))}")

                    latents = model_first_stage.scale_latents(latents)
                    latent_dist = torch.distributions.Normal(model_first_stage.scale_latents(latent_dist.mean).detach().cpu(),
                                                             model_first_stage.scale_latents_std(latent_dist.stddev).detach().cpu())


                    all_latents.append(latents.detach().cpu())
                    all_latent_dists.append(latent_dist)
                    if i * conf.training.batch_size >= n_latents:
                        break

            all_latents = torch.cat(all_latents, dim=0)

            # Compute channel wise std
            channel_stds = all_latents.std(dim=(0, 2, 3))
            channel_means = all_latents.mean(dim=(0, 2, 3))

            print(f"channel means: {channel_means} channel stds: {channel_stds}")

            # And check for spatial uniformity
            avg_latent = all_latents.mean(dim=0, keepdim=True)
            avg_latent_std = all_latents.std(dim=0, keepdim=True)

            save_and_log_latents(avg_latent, "Latents/latent_avg", runPath, 0, run)
            save_and_log_latents(avg_latent_std, "Latents/latent_std", runPath, 0, run)

            # check MMD with gaussian
            mmd_dist = MMDGaussianLoss(sigma=0.1)
            mmd_loss = mmd_dist(all_latents)
            print(f"MMD loss of latents: {mmd_loss.item()}")

            # Compute a KL divergence between the empirical MoG Latent distribution and a Gaussian
            mix = torch.distributions.Categorical(torch.ones(all_latents.shape[0], device=device))
            comp = torch.distributions.Normal(torch.cat([d.mean for d in all_latent_dists], dim=0),
                                                torch.cat([d.stddev for d in all_latent_dists], dim=0))
            comp = torch.distributions.Independent(comp, 3)
            mix_dist = torch.distributions.MixtureSameFamily(mix, comp)

            log_q = mix_dist.log_prob(all_latents)
            log_p = torch.distributions.Normal(0, 1).log_prob(all_latents).mean(dim=(1, 2, 3))

            kl_div = torch.mean(log_q - log_p)
            print(f"KL divergence between empirical MoG and Gaussian: {kl_div.item()}")

            wandb.log({"latent_mmd": mmd_loss.item(), "latent_kl_div": kl_div.item()}, step=0)


        def compute_latent_metrics(dataloaders,
                                   subepoch,
                                   itercount,
                                   epochcount,
                                   metrics=["fid", "kid", "mmd"],
                                   folder_name=None,
                                   gen_samples=True,
                                   compute_refs=False,
                                   ref_feat_path=None,
                                   **generate_kwargs):

            metric_outs = {met: 0 for met in metrics}

            def compute_latent_feats(dataloader):
                if enable_pbars:
                    pbar = tqdm.tqdm(enumerate(dataloader),
                                     total=len(dataloader),
                                     position=rank if data_parallel else 0)
                else:
                    pbar = enumerate(dataloader)

                inception_feats = []
                mmd_feats = []  # simply flattened latents

                fid_resize = Resize((299, 299), interpolation=InterpolationMode.BICUBIC)
                fid_model = pyiqa.create_metric("fid", device=device).net.model

                with (torch.no_grad(), ema.ema_scope("Epoch test")):
                    for i, dataT in pbar:
                        if return_class:
                            data, classes = move_data_to_device(dataT, device=device)
                        else:
                            data = move_data_to_device(dataT, device=device)

                        latents = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode,
                                                   sample=sample,
                                                    cls_tokens=False)
                        latents = model_first_stage.scale_latents(latents)
                        n_chans = latents.shape[1]
                        batch_size = latents.shape[0]

                        if "mmd" in metrics:
                            latents_flat = latents.reshape(batch_size, -1)
                            mmd_feats.append(latents_flat.detach().cpu())

                        if "fid" in metrics or "kid" in metrics:
                            latents = latents / 2  # scale lower so more of the values are within [-1, 1]
                            latents = fid_resize(latents)
                            latents_grey = latents.reshape(-1, *latents.shape[2:]).unsqueeze(1)
                            latents_grey = latents_grey.expand((-1, 3, -1, -1))

                            feat = fid_model(latents_grey.to(device), False, False)
                            feat = feat[0].squeeze(-1).squeeze(-1).detach().cpu()
                            feat = feat.reshape(batch_size, n_chans, -1)
                            inception_feats.append(feat)

                    return torch.cat(inception_feats, dim=0), torch.cat(mmd_feats, dim=0)

            if compute_refs:
                ref_incep_feats, ref_mmd_feats = compute_latent_feats(dataloaders["train_loader"])
                save_path = Path(runPath) / "ref_latent_feats.pt"
                torch.save({"inception": ref_incep_feats, "mmd": ref_mmd_feats}, save_path)
            else:
                assert ref_feat_path is not None
                ref_feats = torch.load(ref_feat_path, map_location="cpu")
                ref_incep_feats, ref_mmd_feats = ref_feats["inception"], ref_feats["mmd"]

            if gen_samples:
                tempdirs = generate(dataloaders, subepoch, itercount, epochcount, folder_name=folder_name, save_latents=True, **generate_kwargs)
            else:
                tempdirs = {"latent": f"{folder_name}_latent"} if folder_name else {"latent": os.path.join(runPath, 'temp_fid_latent')}

            gen_dataset = get_dataset("latents", latent_path=tempdirs["latent"], split="train", modality=["latent"])
            gen_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=conf.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            print(f"found {len(gen_dataset)} generated latents")

            gen_incep_feats, gen_mmd_feats = compute_latent_feats(gen_dataloader)

            if "fid" in metrics:
                metric_outs["fid"] = []  # per channel
                for c in range(gen_incep_feats.shape[1]):
                    mu1, sig1 = np.mean(ref_incep_feats[:, c].numpy(), axis=0), np.cov(ref_incep_feats[:, c].numpy(), rowvar=False)
                    mu2, sig2 = np.mean(gen_incep_feats[:, c].numpy(), axis=0), np.cov(gen_incep_feats[:, c].numpy(), rowvar=False)
                    metric_outs["fid"].append(frechet_distance(mu1, sig1, mu2, sig2))

            if "kid" in metrics:
                metric_outs["kid"] = []  # per channel
                for c in range(gen_incep_feats.shape[1]):
                    metric_outs["kid"].append(kernel_distance(ref_incep_feats[:, c].numpy(), gen_incep_feats[:, c].numpy()))

            if "mmd" in metrics:
                metric_outs["mmd"] = kernel_distance(ref_mmd_feats.numpy(), gen_mmd_feats.numpy())

            if enable_wandb:
                for met, score in metric_outs.items():
                    if isinstance(score, list):
                        for c, s in enumerate(score):
                            wandb.log({f"{met}_latent/channel_{c}": s, "epoch": epochcount, "subepoch": subepoch}, step=itercount)
                    else: 
                        wandb.log({f"{met}_latent": score, "epoch": epochcount, "subepoch": subepoch}, step=itercount)

            return metric_outs


        def compute_manifold_dimensionality(dataloaders, n_samples=72, t=50, n_reps=6*16*16):
            if conf.training.batch_size != 1:
                train_loader, test_loader = create_dataloaders(dataloaders["train_dataset"],
                                                               dataloaders["train_dataset"],
                                                               1,
                                                               world_size, rank,
                                                               data_parallel)
            else:
                train_loader, test_loader = dataloaders["train_loader"], dataloaders["test_loader"]

            n_reps = 6*16*16

            t = torch.ones(conf.training.batch_size, device=device, dtype=torch.long) * t
            spectrums = []

            with (torch.no_grad(), ema.ema_scope("Epoch test")):
                for i, dataT in enumerate(test_loader):
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    latents = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode, sample=False, cls_tokens=False)
                    latents = model_first_stage.scale_latents(latents)
                    latent_shape = latents.shape[1:]

                    latents = latents.expand(conf.training.batch_size, -1, -1, -1)

                    epsilons = []

                    for j in range(math.ceil(n_reps / conf.training.batch_size) + 1):
                        latents_noisy = model.q_sample(latents, t, noise=None)
                        epsilon = model.model(latents_noisy, t)
                        epsilons.append(epsilon.detach().cpu())

                        if j == 0:
                            save_and_log_latents(latents_noisy, f'Latents/noised', runPath, i, run)

                    epsilons = torch.cat(epsilons, dim=0)
                    epsilons = epsilons.reshape(epsilons.shape[0], -1)
                    U, S, Vh = torch.linalg.svd(epsilons)
                    spectrums.append(S)

                    fig, ax = plt.subplots()
                    ax.plot(S)
                    ax.set_ylim(0, S.mean() * 5)
                    ax.set_xlabel("Singular Value Index")
                    ax.set_ylabel("Singular Value")

                    dim = torch.numel(latents[0]) - torch.argmax(torch.abs(torch.diff(S))) - 1
                    print(f"Manifold dimensionality for sample {i}: {dim.item()}")

                    V_samples = Vh[torch.arange(0, n_reps, 100)].reshape(-1, *latent_shape)
                    save_and_log_latents(V_samples, f'singular_vectors', runPath, i, run)

                    if enable_wandb:
                        wandb.log({"manifold_dimensionality": dim.item(), "singular_values": fig}, step=i)

                    if i >= n_samples:
                        break

            spectrums = torch.stack(spectrums, dim=0)
            fig, ax = plt.subplots()
            ax.plot(spectrums.T)
            ax.set_xlabel("Singular Value Index")
            ax.set_ylabel("Singular Value")

            if enable_wandb:
                wandb.log({"all_singular_values": fig}, step=i+1)




        def training_loop(dataloaders):
            best_test_loss = 10000
            test_worsening_count = 0

            print("Entering training loop")
            with Timer('MMVAEplus') as t:
                if conf.training.precompute_latents and not already_precomputed:
                    # Compute the latents and reload the latent dataloader
                    print("Beginning latent computation and saving")
                    precompute_latents(dataloaders, "train", conf.training.precompute_latents_path, dataset_name=conf.dataset_name)
                    precompute_latents(dataloaders, "val", conf.training.precompute_latents_path, dataset_name=conf.dataset_name)

                    latents_train_path = Path(conf.training.precompute_latents_path + "_train") / conf.dataset_name
                    latents_val_path = Path(conf.training.precompute_latents_path + "_val") / conf.dataset_name

                    # Replace the dataloaders with the precomputed latent loaders
                    train_dataset, test_dataset = load_precomputed_datasets(data_conf_0, latents_train_path, latents_val_path)
                    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, conf.training.batch_size, world_size, rank,
                                                                   data_parallel)

                    dataloaders["train_dataset"] = train_dataset
                    dataloaders["test_dataset"] = test_dataset
                    dataloaders["train_loader"] = train_loader
                    dataloaders["test_loader"] = test_loader

                if conf.training.compute_latent_scale:
                    latent_mean, latent_scale = get_latent_scale_factor(dataloaders, n_batches=100, per_channel=conf.training.latent_scale_per_channel)
                    model_first_stage.latent_mean = torch.nn.Parameter(latent_mean)
                    model_first_stage.latent_scale = torch.nn.Parameter(latent_scale)
                    # print("resaving the first stage model with latent mean and scale")
                    # torch.save(model_first_stage.state_dict(), conf.model.params.first_stage_config.params.ckpt_path)

                print(f"Using a latent mean of {model_first_stage.latent_mean} and latent scale of {model_first_stage.latent_scale}")

                if getattr(model_first_stage, "masked_autoencoder", False):
                    fit_mae_latent_disp(dataloaders, n_batches=5)

                epochcount = 0
                n_iter = conf.training.test_every_iter if "test_every_iter" in conf.training else len(dataloaders["train_loader"])
                train_pbar = tqdm.tqdm(enumerate(dataloaders["train_loader"]), total=len(dataloaders["train_loader"]),
                                       position=rank if data_parallel else 0)

                for subepoch, itercount in enumerate(range(start_iter,
                                                           conf.training.num_epochs * n_iter + start_iter,
                                                           n_iter)):
                    print("============================ starting training =========================")
                    epochcount, train_pbar = train_epoch(dataloaders, subepoch, itercount, epochcount, n_iter, train_pbar)
                    print("============================ finished one epoch =========================")

                    test_loss = test_epoch(dataloaders, subepoch, itercount + n_iter, epochcount)
                    print("============================ finished testing =========================")
                    metrics = test_metrics(dataloaders, subepoch, itercount + n_iter, epochcount)
                    print("============================ finished computing FID =========================")

                    if ray_tune:
                        # todo make this work with iteration training
                        checkpoint_data = {
                            "epoch": itercount,
                            "net_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }
                        with tempfile.TemporaryDirectory(dir="/home/ds3119/ray_tmp") as checkpoint_dir:
                            print("Saving ray tune checkpoint and reporting")
                            data_path = Path(checkpoint_dir) / "data.pkl"
                            save_checkpoint(ema, model, optimizer, scheduler, itercount, data_path)
                            checkpoint = Checkpoint.from_directory(checkpoint_dir)
                            train.report({"loss": test_loss, **{f"{met}_{mod}": score for met, scores in metrics.items() for mod, score in zip(modality, scores)}}, checkpoint=checkpoint)
                    else:
                        """
                        print("Checking emas")
                        for i in range(len(ema.emas)):
                            print(i)
                            if torch.allclose(ema.emas[i][test_param_name].data, model.state_dict()[test_param_name].data):
                                print(f"EMA {i} is equal to model parameters at the end of subepoch {subepoch}.")
                            for j in range(i):
                                if torch.allclose(ema.emas[j][test_param_name].data, ema.emas[i][test_param_name].data):
                                    print(f"EMA {j} is equal to EMA {i} parameters at the end of subepoch {subepoch}.")

                        print("checking ema state dict")
                        ema_sd = ema.state_dict()
                        for i in range(len(ema.emas)):
                            print(i)
                            if torch.allclose(ema_sd["emas"][i][test_param_name].data, model.state_dict()[test_param_name].data):
                                print(f"EMA {i} state dict is equal to model parameters at the end of subepoch {subepoch}.")
                            for j in range(i):
                                if torch.allclose(ema_sd["emas"][j][test_param_name].data, ema_sd["emas"][i][test_param_name].data):
                                    print(f"EMA {j} state dict is equal to EMA {i} parameters at the end of subepoch {subepoch}.")
                        """
                        save_checkpoint(ema, model, optimizer, scheduler, itercount+n_iter, runPath / f"model_{itercount+n_iter}.pt")
                        if test_loss < best_test_loss:
                            pass
                        else:
                            test_worsening_count += 1
                            if test_worsening_count > conf.training.patience and not ray_tune:
                                print(
                                    f"Early stopping. Best test loss attained in epoch: {subepoch - conf.training.patience}")
                                logger.info("saving last model")
                                save_checkpoint(ema, model, optimizer, scheduler, itercount+n_iter, runPath / "last_model.pt")

                                break

        if conf.function.name is not None:
            locals()[conf.function.name](dataloaders=dataloaders, **conf.function.kwargs)
        else:
            training_loop()

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()

        if not conf.training.keep_intermediate_models:
            models = [s for s in glob.glob(f"{str(runPath)}/model_*.pt")]
            model_epochs = [int(os.path.basename(m).split("_")[1].split(".")[0]) for m in models]

            models = sorted(models, key=lambda x: model_epochs[x])
            for m in models[:-1]:
                os.remove(m)




def run_training_ray_tune(config, conf, **params):
    conf = override_config(conf, config)
    print(conf)

    run_training(conf=conf, **params)


@hydra.main(version_base=None, config_path="configs", config_name="nyu_depth_lsun_stable_diffusion_train.yaml")
def main(conf: DictConfig) -> None:
    conf = expand_sub_confs(conf)

    # Random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)


    if optimise_hyperparameters:
        logger.info("Beginning hyperparameter tuning")
        ray.init(_temp_dir="/home/ds3119/ray_tmp")

        num_samples = 10
        max_num_epochs = 15
        gpus_per_trial = 1

        output_dir = Path(os.path.join(conf.outputdir, conf.experiment, "ray_tune"))
        output_dir.mkdir(parents=True, exist_ok=True)

        assert isinstance(conf.device, str), "Hyperparameter optimisation only works with single GPU"
        config = {
            "model.params.unet_config.params.model_channels": tune.choice([128, 192, 320, 448]),
            "model.params.unet_config.params.num_res_blocks": tune.choice([1, 2, 3]),
            "training.lr": tune.choice([ 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            # "training.batch_size": tune.choice([8, 16, 32, 64]),
        }

        algo = TuneBOHB(metric="loss", mode="min")
        algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=1)
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            metric="loss",
            mode="min",
            max_t=1800)

        trainable = partial(run_training_ray_tune, rank="cuda:0", conf=conf, data_parallel=False, world_size=1, ray_tune=True, enable_pbars=False)
        trainable_with_resources = tune.with_resources(trainable,
                                                       resources={"gpu": 1})

        tuner = tune.Tuner(
            trainable_with_resources,
            tune_config=tune.TuneConfig(
                search_alg=algo,
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            run_config=train.RunConfig(
                name="bohb_exp",
                stop={"training_iteration": max_num_epochs},
            ),
            param_space=config,
        )
        result = tuner.fit()

        # save last results
        result.get_dataframe().to_csv(output_dir / "last_epoch_results.csv")

        for metric in ["loss", "fid_colour", "fid_depth"]:
            print("==================")
            print(f"{metric}")
            print("==================")
            best_trial = result.get_best_result(metric, "min")
            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final validation loss: {best_trial.metrics['loss']}")

            best_checkpoint = best_trial.checkpoint
            with best_checkpoint.as_directory() as dir:
                data_path = Path(dir) / "data.pkl"
                shutil.copy(data_path, output_dir / "best_checkpoint_loss_data.pkl")

            print("==================")

            # Get last reported results per trial
            df_best_metric = result.get_dataframe(
                filter_metric=metric, filter_mode="min"
            )
            df_best_metric.to_csv(output_dir / f"best_{metric}_results.csv")
    else:
        # start the training function, possibly with data parallelism
        if isinstance(conf.device, str):
            run_training(conf.device, conf, False)
        elif isinstance(conf.device, Iterable):
            mp.spawn(
                run_training,
                args=(conf, True, len(conf.device)),
                nprocs=len(conf.device),
                join=True
            )


if __name__ == '__main__':
    main()

