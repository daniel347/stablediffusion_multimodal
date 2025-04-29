# Train MMVAEplus on CUB Image-Captions dataset
import os
import shutil
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
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
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler, ExponentialLR, CosineAnnealingWarmRestarts, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.utils as tvu
from torch.amp import GradScaler, autocast

import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ray.cloudpickle as pickle
import tempfile
import sys
import glob
import logging

from ldm.util import instantiate_from_config, Timer, save_model_light, GradAutoClipper
from ldm.data import get_dataset
from ldm.modules.ema import DummyEMA
import pyiqa
import cmmd_pytorch as cmmd
from utils import save_and_log_latents, save_and_log_img, expand_sub_confs, duplicate_stable_diff_weights, move_data_to_device, override_config
from reconstruct_model_phema import reconstruct_phema
import ldm.models.diffusion as diffusion

import random
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

enable_wandb = True
optimise_hyperparameters = False
log_iterations = True

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

notes = ("Train diffusion with 32x32 latents. THe semantic segmentation autoencoder reconstruction is poor, "
         "so we dont expect brilliant results, but if we avoid unrealistic RGB generations this would indicate "
         "diffusion prefers higher resolution latents")
name = "32_latents"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def load_precomputed_datasets(conf):
    train_dataset = get_dataset("latents", latent_folder_name=conf.training.precompute_latents_path, split="train", **conf.data)
    test_dataset = get_dataset("latents", latent_folder_name=conf.training.precompute_latents_path, split="val", **conf.data)

    return train_dataset, test_dataset


def save_checkpoint(ema, model, optimiser, scheduler, iteration, path):
    sd = {"ema": ema.state_dict(),
          "model": model.state_dict(),
          "optimiser": optimiser.state_dict(),
          "scheduler": scheduler.state_dict(),
          "iteration": iteration}

    torch.save(sd, path)


def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr


def load_checkpoint(ema, model, optimiser, scheduler, path):
    sd = torch.load(path)
    model.load_state_dict(sd["model"])
    if "ema" in sd:
        ema.load_state_dict(sd["ema"])
    else:
        logger.warning("No EMA state dict found in checkpoint")

    if "optimiser" in sd:
        optimiser.load_state_dict(sd["optimiser"])
    else:
        logger.warning("No optimiser state dict found in checkpoint")

    if "scheduler" in sd:
        scheduler.load_state_dict(sd["scheduler"])
    else:
        logger.warning("No scheduler state dict found in checkpoint")

    return sd["iteration"]


def load_image_datasets(conf):
    tx = v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)])
    return_class = "return_class" in conf.data and conf.data.return_class
    return_index = conf.training.precompute_latents

    train_opts = conf.data
    test_opts = copy.deepcopy(OmegaConf.to_container(conf.data))
    for key in ["p_hor_flip", "p_colour_jitter"]:
        test_opts.pop(key, None)

    train_dataset = get_dataset(conf.data.dataset_name,
                                transform=tx,
                                split="train",
                                return_index=return_index,
                                return_class=return_class,
                                **train_opts)
    test_dataset = get_dataset(conf.data.dataset_name,
                               transform=tx,
                               split="val",
                               return_index=return_index,
                               return_class=return_class,
                               **test_opts)

    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, conf, world_size=1, rank=0, data_parallel=False):
    kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                       shuffle=True) if data_parallel else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank,
                                      shuffle=False) if data_parallel else None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.training.batch_size,
                                               shuffle=not data_parallel, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.training.batch_size, shuffle=False,
                                              sampler=test_sampler, **kwargs)

    return train_loader, test_loader


def initialise_model(conf):
    OmegaConf.to_container(conf["model_conf"]["model"])
    model = instantiate_from_config(conf["model_conf"]["model"])

    if conf.init_from_stable_diffusion:
        # sd_state_dict = torch.load("checkpoints/sd-v1-4.ckpt")
        model.init_from_ckpt("checkpoints/v2-1_512-ema-pruned.ckpt")

        """
        u_net_dict = {}
        for name, param in sd_state_dict["state_dict"].items():
            if "model.diffusion_model." in name:
                new_name = name.split("model.diffusion_model.")[1]
                u_net_dict[new_name] = param

        M, U = model.model.load_state_dict(u_net_dict, strict=False)
        for m in M:
            print(f"Missing key: {m}")
        for u in U:
            print(f"Unused key: {u}")

        first_stage_dict = {}
        for name, param in sd_state_dict["state_dict"].items():
            if "first_stage_model" in name:
                new_name = name.split("first_stage_model.")[1]
                first_stage_dict[new_name] = param

        for vae in model.vaes:
            ignore_keys = ["encoder.conv_in", "decoder.conv_out"] if vae.is_semseg else []
            ignore_keys += ["encoder.conv_out", "quant_conv", "decoder.conv_in", "post_quant_conv"] if \
            vae.decoder.z_shape[1] != 4 else []
            sd = copy.deepcopy(first_stage_dict)
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]

            # If we have doubled the number of channels we must go though and double all the initialisation weight shapes
            sd = duplicate_stable_diff_weights(sd, vae)

            M, U = vae.load_state_dict(sd, strict=False)
            for m in M:
                print(f"Missing key: {m}")
            for u in U:
                print(f"Unused key: {u}")
        """

    return model


def cat_dists_or_tensor_latents(latents):
    if all([isinstance(l, torch.Tensor) for l in latents]):
        # Tensors
        latents = torch.cat(latents, dim=1)
    else:
        # Distributions
        latents = torch.cat([l.mean for l in latents], dim=1)
    return latents


def generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode):
    avg_z = z_mode == "avg"
    random_z = z_mode == "random"
    channel_u = (model_first_stage.w_channels * NUM_VAES + model_first_stage.z_channels) if model_first_stage.factorised_latent_space else model_first_stage.u_channels

    if conf.training.precompute_latents:
        latents = data
    else:
        latents = model_first_stage.encode(data, avg_z=avg_z)

    if isinstance(latents, torch.Tensor):
        if latents.shape[1] == channel_u:
            return latents
        else:
            latents = torch.split(latents, [model_first_stage.w_channels for _ in range(NUM_VAES)] + [
            model_first_stage.z_channels for _ in range(NUM_VAES)], dim=1)
        
    if isinstance(latents, list):
        if len(latents) > NUM_VAES + 1:
            assert (random_z or isinstance(z_mode, int)), "One of average or random z must be set or a integer supplied to choose one z, or the z-combiner must be used"
            latents = [*latents[:NUM_VAES], random.choice(latents[NUM_VAES:])]
            return cat_dists_or_tensor_latents(latents)

        else:
            return cat_dists_or_tensor_latents(latents)


def run_training(rank, conf, data_parallel=False, world_size=None, ray_tune=False, enable_pbars=True):
    try:
        if data_parallel:
            setup(rank, world_size)

        print("Working directory: ", os.getcwd())

        use_mixed_precision = conf.training.use_mixed_precision
        conf["model_conf"]["model"]["params"]["unet_config"]["params"]["use_fp16"] = use_mixed_precision
        if use_mixed_precision:
            print("Using mixed precision training")
            scaler = GradScaler()

        # Initialise the model
        if "ckpt_path" not in conf.model_conf.model.params.first_stage_config.params or \
                conf.model_conf.model.params.first_stage_config.params.ckpt_path is None:
            logger.warning("No checkpoint path for first stage! This is could be unintentional")
        model = initialise_model(conf)
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

        # Set experiment name if not set
        if not conf.experiment:
            conf.experiment = model.modelName

        # Set up run path
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        if conf.training.test_only:
            runId += "_test"

        if ray_tune:
            runId += f"_rank{train.get_context().get_world_rank()}"

        if name is not None:
            runId += f"_{name}"

        if rank == 0 or not data_parallel:
            experiment_dir = Path(os.path.join(conf.data.outputdir, conf.experiment, "checkpoints"))
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
                notes=notes,
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
                                      CosineAnnealingWarmRestarts(optimizer, 4000, 1, eta_min=0.1 * conf.training.lr)])
                                      # ExponentialLR(optimizer, conf.training.gamma)])


        """
        scheduler = LambdaLR(optimizer, lambda epoch: learning_rate_schedule(epoch * conf.training.batch_size,
                                                                             conf.training.batch_size,
                                                                             ref_lr=conf.training.lr,
                                                                             ref_batches=1000,
                                                                             rampup_Mimg=0.005))
        """

        scheduler_step_freq = 1

        start_epoch = None


        if conf.model_initialisation is not None:
            start_epoch = load_checkpoint(ema, model, optimizer, scheduler, conf.model_initialisation)


        if ray_tune:
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    start_epoch = load_checkpoint(ema, model, optimizer, scheduler, data_path)

        if start_epoch is None:
            start_epoch = conf.training.start_epoch

        train_dataset, test_dataset = load_image_datasets(conf)
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, conf, world_size, rank, data_parallel)

        display_data_fs = [getattr(train_dataset, f"{mode}_display") for mode in conf.data.modality]

        already_precomputed = False
        if conf.training.precompute_latents:
            latents_path = Path(conf.data.root_data_dir) / Path(conf.training.precompute_latents_path)
            if latents_path.exists() and not conf.training.force_recompute:
                # We have already computed the latents
                train_dataset, test_dataset = load_precomputed_datasets(conf)
                train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, conf, world_size, rank,
                                                               data_parallel)
                already_precomputed = True

        return_class = "return_class" in conf.data and conf.data.return_class
        return_index = conf.training.precompute_latents
        modalities = conf.data.modality
        avg_z = conf.training.z_mode == "avg"
        random_z = conf.training.z_mode == "random"
        z_combiner = conf.training.z_mode == "z_combiner"
        
        if z_combiner:
            model_first_stage.use_z_comb_network = True

        # Set the learning rate forward to continue training after pause
        if conf.training.start_epoch > 0:
            for _ in range((len(train_loader) * conf.training.start_epoch)//scheduler_step_freq):
                scheduler.step()

        def get_latent_scale_factor(split="train", n_batches=1, per_channel=True):
            data_loader = train_loader if split == "train" else test_loader

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader),
                                 position=rank if data_parallel else 0)
            else:
                pbar = enumerate(data_loader)

            channel_means = 0
            channel_vars = 0

            with torch.no_grad():
                for i, dataT in pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    latents = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode)

                    if per_channel:
                        channel_means += latents.mean(dim=(0, 2, 3))
                        channel_vars += latents.var(dim=(0, 2, 3))
                    else:
                        channel_means += latents.mean(dim=(0, 1, 2, 3))
                        channel_vars += latents.var(dim=(0, 1, 2, 3))

                    if i >= n_batches:
                        break

            channel_means = channel_means / n_batches
            channel_stds = torch.sqrt(channel_vars / n_batches)

            return channel_means, channel_stds

        def precompute_latents(split="train", save_folder="latents"):
            dataloader = train_loader if split == "train" else test_loader
            model.eval()

            if data_parallel:
                dataloader.sampler.set_epoch(0)

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                                 position=rank if data_parallel else 0)
            else:
                pbar = enumerate(dataloader)

            save_dir = Path(conf.data.root_data_dir) / Path(f"{save_folder}/{split}")

            with torch.no_grad():
                for i, dataT in pbar:
                    if return_class:
                        data, classes, ind = move_data_to_device(dataT, device=device)
                    else:
                        data, ind = move_data_to_device(dataT, device=device)
                    # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                    optimizer.zero_grad()

                    if use_mixed_precision:
                        with autocast(dtype=torch.float16, device_type="cuda"):
                            latents = model_first_stage.encode(data, avg_z=avg_z)
                    else:
                        latents = model_first_stage.encode(data, avg_z=avg_z)

                    if isinstance(latents, list):
                        latents = torch.cat(latents, dim=1)

                    print(latents.shape)

                    for j, latent in enumerate(latents):
                        if return_class:
                            save_dir_class = save_dir / Path(str(classes[j].item()))
                        else:
                            save_dir_class = save_dir / Path("all")

                        if not save_dir_class.exists():
                            save_dir_class.mkdir(parents=True)

                        save_name = save_dir_class / Path(f"latent_{ind[j].item()}.pt")
                        torch.save(latent.cpu(), save_name)

        def train_epoch(epoch):
            model.train()
            b_loss = 0

            if data_parallel:
                train_loader.sampler.set_epoch(epoch)

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=rank if data_parallel else 0)
            else:
                pbar = enumerate(train_loader)

            for i, dataT in pbar:
                if return_class:
                    data, classes = move_data_to_device(dataT, device=device)
                else:
                    data = move_data_to_device(dataT, device=device)
                # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                optimizer.zero_grad()

                latents = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode)
                latents = model_first_stage.scale_latents(latents)

                # print(f"Latent scale: {latents.std(dim=(0, 2, 3))}   mean: {latents.mean(dim=(0, 2, 3))}")

                if use_mixed_precision:
                    with autocast(dtype=torch.float16, device_type="cuda"):
                        loss, loss_dict = model(latents, c=(classes if return_class else None), t_sampler=t_sampler)

                    scaler.scale(loss).backward()

                    if grad_clipper is not None:
                        scaler.unscale_(optimizer)
                        grad_clipper.autoclip_grad()
                    scaler.step(optimizer)
                else:
                    loss, loss_dict = model(latents, c=(classes if return_class else None), t_sampler=t_sampler)
                    loss.backward()

                    if grad_clipper is not None:
                        grad_clipper.autoclip_grad()
                    optimizer.step()

                if torch.isnan(loss).item():
                    if enable_wandb:
                        wandb.alert(title="Loss is NaN", text=f"Halting training at epoch {epoch} and step {i}")
                    raise ValueError("Loss has become NaN, halting training")

                b_loss += loss.detach().item()
                if conf.print_freq > 0 and i % conf.print_freq == 0 and enable_pbars:
                    pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))

                if ema is not None:
                    ema.update(conf.training.batch_size)

                if i % scheduler_step_freq == 0:
                    scheduler.step()
                    # print(f"Learning rate: {scheduler.get_last_lr()}")

                step = epoch * len(train_loader) + i if log_iterations else epoch

                if enable_wandb and log_iterations:
                    wandb.log({"Loss/train": loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch}, step=step)

                if i == 0:
                    if conf.training.precompute_latents:
                        if model_first_stage.factorised_latent_space and False:
                            *latents_w, latents_z = torch.split(latents, [model_first_stage.w_channels for _ in range(NUM_VAES)] + [model_first_stage.z_channels], dim=1)
                            for j, latent in enumerate(latents_w):
                                save_and_log_latents(latent, f'Latents/wm{j}', runPath, step, run)
                            if latents_z.shape[1] > 0:
                                save_and_log_latents(latents_z, f'Latents/z', runPath, step, run)
                        else:
                            save_and_log_latents(latents, f'Latents/u', runPath, step, run)
                    else:
                        for m, (name, dbatch) in enumerate(zip(modalities, data)):
                            save_and_log_img(dbatch, f"Data/{name}", runPath,
                                             display_data_fs[m], step=step, wandb_run=run)

            # Epoch loss
            epoch_loss = b_loss / len(train_loader.dataset)
            if enable_wandb and not log_iterations: wandb.log({"Loss/train": epoch_loss}, step=epoch)
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))

        def test_epoch(epoch, reencode_end=False):
            step = (epoch+1) * len(train_loader) if log_iterations else epoch
            if "sampler" in conf:
                run_params = conf.sampler.run_params
            else:
                run_params = {"S": 1000}  # For all but DDPM we must specify a number of timesteps, so we default to 1000

            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0)
            else:
                pbar = enumerate(test_loader)

            b_loss = 0
            with (torch.no_grad(), ema.ema_scope("Epoch test")):
                for i, dataT in pbar:
                    if return_class:
                        data, classes = move_data_to_device(dataT, device=device)
                    else:
                        data = move_data_to_device(dataT, device=device)

                    latents = generate_latents(conf, model_first_stage, data, NUM_VAES, z_mode=conf.training.z_mode)
                    latents = model_first_stage.scale_latents(latents)

                    loss, loss_dict = model(latents, c=(classes if return_class else None), t_sampler=t_sampler)
                    b_loss += loss.detach().item()

                    if i == 0 and epoch % 1 == 0:
                        # Perform test generations

                        N_gen = 16
                        if return_class:
                            classes = torch.randint(low=0,
                                                    high=conf.model_conf.model.params.unet_config.params.num_classes,
                                                    size=(N_gen,),
                                                    device=device)

                        samples_zw = sampler.sample(verbose=enable_pbars,
                                                    batch_size=N_gen,
                                                    shape=model.input_shape,
                                                    conditioning=(classes if return_class else None),
                                                    **run_params)

                        # print(f"Generated latent scale: {samples_zw.std(dim=(0, 2, 3))}   mean: {samples_zw.mean(dim=(0, 2, 3))}")

                        samples_zw_scaled = model_first_stage.scale_latents(samples_zw, reverse=True)
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
                                                 runPath, display_data_fs[m], step, run, wandb_kwargs=wandb_kwargs,
                                                 save_kwargs={}, nrow=4)
                            save_and_log_latents(samples_zw_2, f'Generation/reencoded_latent',
                                                 runPath, step, run, wandb_kwargs=wandb_kwargs)



                        for m in range(len(samples_x)):
                            save_and_log_img(samples_x[m], f'Generation/m{m}',
                                             runPath, display_data_fs[m], step, run, wandb_kwargs=wandb_kwargs, save_kwargs={}, nrow=4)

                        save_and_log_latents(samples_zw, f'Generation/latent',
                                         runPath, step, run, wandb_kwargs=wandb_kwargs)

            # Epoch test loss
            epoch_loss = b_loss / len(test_loader.dataset)
            if enable_wandb: wandb.log({"Loss/test": epoch_loss}, step=step)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))

            return epoch_loss

        def test_metrics(epoch, n_samples=32, use_diffusion=True, reencode_end=False, metrics=["fid", "kid", "cmmd"]):
            if "sampler" in conf:
                run_params = conf.sampler.run_params
            else:
                run_params = {"S": 1000}  # For all but DDPM we must specify a number of timesteps, so we default to 1000

            step = ((epoch+1) * len(train_loader)) if log_iterations else epoch

            gen_batch_size = 16
            n_batches = n_samples // gen_batch_size

            tempdirs = {}
            for mod in modalities:
                tempdir = os.path.join(runPath, f'temp_fid_{mod}')
                os.makedirs(tempdir, exist_ok=True)
                tempdirs[mod] = tempdir

            if enable_pbars:
                pbar = tqdm.tqdm(range(n_batches))
            else:
                pbar = range(n_batches)

            model.eval()
            with torch.no_grad():
                for n in pbar:
                    if use_diffusion:
                        if return_class:
                            classes = torch.randint(low=0,
                                                    high=conf.model_conf.model.params.unet_config.params.num_classes,
                                                    size=(gen_batch_size,),
                                                    device=device)

                        latents = sampler.sample(verbose=enable_pbars,
                                                 batch_size=gen_batch_size,
                                                 shape=model.input_shape,
                                                 conditioning=(classes if return_class else None),
                                                 **run_params)

                        latents = model_first_stage.scale_latents(latents, reverse=True)
                    else:
                        latent_dim = (
                        gen_batch_size, *(model.module.input_shape if data_parallel else model.input_shape))
                        latents = torch.randn(latent_dim, device=device)
                    samples_x = model_first_stage.decode(latents)

                    if reencode_end:
                        samples_zw = model_first_stage.encode([sx.mean for sx in samples_x], avg_z=avg_z)
                        samples_zw = cat_dists_or_tensor_latents(samples_zw)
                        samples_x = model_first_stage.decode(samples_zw)

                    for m, (name, samples) in enumerate(zip(modalities, samples_x)):
                        for i, s in enumerate(samples.mean):
                            s_disp = display_data_fs[m](s.detach().cpu())
                            tvu.save_image(s_disp, os.path.join(tempdirs[name], f"sample_{name}_{n * gen_batch_size + i}.png"))

            metric_scores = {}
            for met in metrics:
                if met == "cmmd":
                    metric = partial(cmmd.compute_cmmd, device=device)
                else:
                    metric = pyiqa.create_metric(met, device=device)

                scores = []
                for mod in modalities:
                    if mod in ["colour", "depth"]:
                        split = "test" if conf.data.dataset_name == "nyu_depth" else "val"
                        score = metric(tempdirs[mod], dataset_name=f"{conf.data.dataset_name}_{mod}", dataset_res=299, dataset_split=split, center_crop=True)
                        if enable_wandb:
                            wandb.log({f"{met}_{mod}": score, "epoch": epoch}, step=step)

                        scores.append(score)

                        print(f'====>             {met} {mod}: {score:.2f}')
                    else:
                        scores.append(None)

                metric_scores[met] = scores

            return metric_scores

        def test_optimum_EMA(out_epochs, out_stds):
            saves_path = os.path.dirname(conf.model_initialisation)

            out_epochs, out_stds = zip(*product(out_epochs, out_stds))

            model_names = [f"model_it{out_es}_ema_{out_std:.3f}.pt" for out_es, out_std in zip(out_epochs, out_stds)]

            it = reconstruct_phema(saves_path, out_std=out_stds, out_nepoch=out_epochs, out_dir=saves_path, epoch_size=24000)
            for _ in it:
                pass

            for step, name in enumerate(model_names):
                logging.info(f"Testing configuration with epochs:{out_epochs[step]} and std:{out_stds[step]}")
                load_checkpoint(ema, model, optimizer, scheduler, os.path.join(saves_path, name))
                test_epoch(step)
                test_metrics(step)

                log_step = ((step + 1) * len(train_loader)) if log_iterations else step

                if enable_wandb:
                    wandb.log({"n_epochs": out_epochs[step], "ema_std": out_stds[step]}, step=log_step)


        def plot_timestep_logvars():
            logvars = model.logvar.cpu().detach().numpy()
            timesteps = np.arange(0, len(logvars))
            plot = sns.lineplot(x=timesteps, y=logvars)
            plot.set(xlabel="Diffusion timestep", ylabel="timestep log weighting")
            fig = plot.get_figure()
            fig.savefig(os.path.join(runPath, "timestep_logvars.png"))


        def training_loop():
            best_test_loss = 10000
            test_worsening_count = 0

            print("Entering training loop")
            with Timer('MMVAEplus') as t:
                if conf.training.precompute_latents and not already_precomputed:
                    # Compute the latents and reload the latent dataloader
                    print("Beginning latent computation and saving")
                    precompute_latents("train", conf.training.precompute_latents_path)
                    precompute_latents("val", conf.training.precompute_latents_path)

                    # Replace the dataloaders with the precomputed latent loaders
                    train_dataset, test_dataset = load_precomputed_datasets(conf)
                    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, conf, world_size, rank, data_parallel)


                if conf.training.compute_latent_scale:
                    latent_mean, latent_scale = get_latent_scale_factor(n_batches=100, per_channel=conf.training.latent_scale_per_channel)
                    model_first_stage.latent_mean = torch.nn.Parameter(latent_mean)
                    model_first_stage.latent_scale = torch.nn.Parameter(latent_scale)
                    print("resaving the fisrt stage model with latent mean and scale")
                    torch.save(model_first_stage.state_dict(), conf.model_conf.model.params.first_stage_config.params.ckpt_path)

                print(f"Using a latent mean of {model_first_stage.latent_mean} and latent scale of {model_first_stage.latent_scale}")

                # metrics = test_metrics(0)
                # print(metrics)
                # fid_scores = test_fid(0, n_samples=256)
                # sys.exit()

                # test_epoch(start_epoch)
                # fid_scores = test_fid(start_epoch)
                # print("   ".join([f"{mod} FID: {fid}" for mod, fid in zip(modalities, fid_scores)]))
                for epoch in range(start_epoch + 1, conf.training.num_epochs + 1 + start_epoch):
                    print("============================ starting training =========================")
                    train_epoch(epoch)
                    print("============================ finished one epoch =========================")

                    test_loss = test_epoch(epoch)
                    print("============================ finished testing =========================")
                    metrics = test_metrics(epoch)
                    print("============================ finished computing FID =========================")

                    if ray_tune:
                        checkpoint_data = {
                            "epoch": epoch,
                            "net_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }
                        with tempfile.TemporaryDirectory(dir="/home/ds3119/ray_tmp") as checkpoint_dir:
                            print("Saving ray tune checkpoint and reporting")
                            data_path = Path(checkpoint_dir) / "data.pkl"
                            save_checkpoint(ema, model, optimizer, scheduler, epoch, data_path)
                            checkpoint = Checkpoint.from_directory(checkpoint_dir)
                            train.report({"loss": test_loss, **{f"{met}_{mod}": score for met, scores in metrics.items() for mod, score in zip(modalities, scores)}}, checkpoint=checkpoint)
                    else:
                        save_checkpoint(ema, model, optimizer, scheduler, epoch, runPath / f"model_{epoch}.pt")
                        """
                        best_test_loss = test_loss
                        test_worsening_count = 0
                        if (rank == 0 or not data_parallel) and not ray_tune:
                            save_checkpoint(ema, model, optimizer, scheduler, epoch, runPath / f"model_{epoch}.pt")
                        """
                        if test_loss < best_test_loss:
                            pass
                        else:
                            test_worsening_count += 1
                            if test_worsening_count > conf.training.patience and not ray_tune:
                                print(
                                    f"Early stopping. Best test loss attained in epoch: {epoch - conf.training.patience}")
                                logger.info("saving last model")
                                save_checkpoint(ema, model, optimizer, scheduler, epoch, runPath / "last_model.pt")

                                break

        if conf.function.name is not None:
            locals()[conf.function.name](**conf.function.kwargs)
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


@hydra.main(version_base=None, config_path="configs/training/", config_name="celeba_semseg_diffusion_train_only_rgb.yaml")
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

        output_dir = Path(os.path.join(conf.data.outputdir, conf.experiment, "ray_tune"))
        output_dir.mkdir(parents=True, exist_ok=True)

        assert isinstance(conf.device, str), "Hyperparameter optimisation only works with single GPU"
        config = {
            "model_conf.model.params.unet_config.params.model_channels": tune.choice([128, 192, 320, 448]),
            "model_conf.model.params.unet_config.params.num_res_blocks": tune.choice([1, 2, 3]),
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

