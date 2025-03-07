# Train MMVAEplus on CUB Image-Captions dataset
import os

import pandas as pd
import psutil
import shutil
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
import tqdm
import time
from collections.abc import Iterable
import copy
from datetime import datetime
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

import torch
from torch import optim
from torchvision.transforms import v2, InterpolationMode
from torchvision.utils import make_grid
from torch.profiler import ProfilerActivity, profile, record_function
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler, ExponentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.utils as tvu
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import pandas as pd

from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
from ldm.util import instantiate_from_config, dict_to_namespace, namespace_to_dict, Timer, save_model_light, GradAutoClipper
from mmvae_objectives import m_dreg, train_adversary
from ldm.data import get_dataset

import pyiqa

enable_wandb = False

def expand_sub_confs(conf):
    # Expand sub-configs
    for key, val in conf.items():
        if key[-4:] == "conf":
            subconf = OmegaConf.load(val)
            conf[key] = expand_sub_confs(subconf)
    return conf


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def initialise_model(conf):
    model = instantiate_from_config(conf["model_conf"]["model"]["params"]["first_stage_config"])

    if conf.init_from_stable_diffusion:
        print("Initialising autoencoders with stable diffusion weights")
        sd_state_dict = torch.load("checkpoints/v2-1_512-ema-pruned.ckpt")
        first_stage_dict = {}
        for name, param in sd_state_dict["state_dict"].items():
            if "first_stage_model" in name:
                new_name = name.split("first_stage_model.")[1]
                first_stage_dict[new_name] = param

        for vae in model.vaes:
            ignore_keys = ["encoder.conv_in", "decoder.conv_out"] if vae.is_semseg else []
            sd = copy.deepcopy(first_stage_dict)
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]

            M, U = vae.load_state_dict(sd, strict=False)
            for m in M:
                print(f"Missing key: {m}")
            for u in U:
                print(f"Unused key: {u}")
        model = MMVAEDataParallelWrapper(model)

    elif conf.model_initialisation is not None:
        print("Initialising with custom weights")
        model = MMVAEDataParallelWrapper(model)
        model.load_state_dict(torch.load(conf.model_initialisation))

    return model

def stack_generation_matrices(mats, data, mode_display_fs=None):
    # Given a list of N_gen cross and self generation matrices, get them into a single matix
    n_modes = len(mats[0])
    n_inputs = data[0].shape[0]

    out_mat = [[None for _ in range(n_modes)] for _ in range(n_modes)]
    for i in range(n_modes):
        for j in range(n_modes):
            img_shape = (3, *mats[0][i][j].shape[3:])
            input_data = data[i].detach().cpu()
            output_data = [m[i][j].squeeze(0).detach().cpu() for m in mats]

            if mode_display_fs is not None:
                # display functions dont handle batches so we use list comprehension
                input_data = torch.stack([mode_display_fs[i](inp) for inp in input_data], dim=0)
                output_data = [torch.stack([mode_display_fs[j](o) for o in out], dim=0) for out in output_data]

            out_mat[i][j] = make_grid(torch.cat([input_data] + output_data, dim=0).view(-1, *img_shape),
                                      nrow=n_inputs)

    return out_mat


def run_training(rank, conf, data_parallel=False, world_size=None):
    try:
        if data_parallel:
            setup(rank, world_size)

        # Initialise with SD model?
        model = initialise_model(conf)
        device = rank
        model = model.to(device)

        if data_parallel:
            print(f"Initialising data-based parallelism with {len(conf.device)} GPUs")
            model = DDP(model, device_ids=[rank])


        NUM_VAES = len(model.module.module.vaes) if data_parallel else len(model.module.vaes)

        # Set experiment name if not set
        if not conf.experiment:
            conf.experiment = model.modelName

        # Set up run path
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        if rank == 0 or not data_parallel:
            experiment_dir = Path(os.path.join(conf.data.outputdir, conf.experiment, "checkpoints"))
            experiment_dir.mkdir(parents=True, exist_ok=True)
            runPath = os.path.join(str(experiment_dir), runId)
            if os.path.exists(runPath):
                shutil.rmtree(runPath)
            os.makedirs(runPath)
            print('Expt:', runPath)
            print('RunID:', runId)

            # save args to run
            with open('{}/args.json'.format(runPath), 'w') as fp:
                OmegaConf.save(config=conf, f=fp)

        # WandB
        if enable_wandb:
            wandb.login()
            wandb.init(
                # Set the project where this run will be logged
                project=conf.experiment,
                # Track hyperparameters and run metadata
                config=OmegaConf.to_container(conf),
                # Run name
                name=f"{runId}_{str(rank)}",
                group=f"DDP_{runId}",
            )

            wandb.alert(title="Starting Run", text="Started training stable diffusion autoencoder")

        # preparation for training
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=conf.training.lr, amsgrad=True)

        scheduler = ChainedScheduler([LinearLR(optimizer, 0.1, 1, total_iters=10),
                                      ExponentialLR(optimizer, 0.99)])

        if "discriminator" in conf.training and conf.training.discriminator is not None:
            disc = instantiate_from_config(conf.training.discriminator)
            disc_optimizer = optim.Adam(disc.parameters(), lr=conf.training.discriminator_lr, amsgrad=True)

            train_adversary_f = partial(train_adversary, discriminator=disc, optimiser=disc_optimizer, device=device)
        else:
            train_adversary_f = None
            assert conf.training.wadversary == 0.0

        lpips = pyiqa.create_metric("lpips", device=device, as_loss=True)


        scheduler_step_freq = 50

        use_mixed_precision = conf.training.use_mixed_precision
        if use_mixed_precision:
            print("Using mixed precision training")
            scaler = GradScaler()

        # Load dataset
        modalities = conf.data.modalities
        kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
        tx = {"colour": v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)]),
              "semseg": v2.Compose([v2.Resize(conf.data.input_size, interpolation=InterpolationMode.NEAREST_EXACT),
                                    v2.CenterCrop(conf.data.input_size)])}

        train_dataset = get_dataset(conf.data.dataset_name,
                                    root_data_dir=conf.data.datadir,
                                    transform=tx,
                                    split="train",
                                    modality=modalities,
                                    p_hor_flip=0.5,
                                    p_colour_jitter=0.1)
        test_dataset = get_dataset(conf.data.dataset_name,
                                   root_data_dir=conf.data.datadir,
                                   transform=tx,
                                   split="val",
                                   modality=modalities)

        display_data_fs = [getattr(train_dataset, f"{mode}_display") for mode in modalities]

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                           shuffle=True) if data_parallel else None
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank,
                                          shuffle=False) if data_parallel else None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.training.batch_size,
                                                   shuffle=not data_parallel, sampler=train_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.training.batch_size, shuffle=False,
                                                  sampler=test_sampler, **kwargs)

        if "weight_semseg_classes" in conf.training and conf.training.weight_semseg_classes is not None:
            print("using inverse class frequency weighting for semantic segmentation loss")
            class_freqs = torch.load(os.path.join(conf.data.datadir, conf.training.weight_semseg_classes))
            class_weights = 1.0 / class_freqs
            class_weights /= class_weights.sum()
            class_weights = class_weights.to(device)

            class_weights = [class_weights if m == "semseg" else None for m in modalities]
        else:
            class_weights = [None for _ in modalities]


        grad_clipper = None
        if "grad_clipper" in conf.training:
            grad_clipper = GradAutoClipper(model, **conf.training.grad_clipper)

        # Set the learning rate forward to continue training after pause
        if conf.training.start_epoch > 0:
            for _ in range(len(train_loader)//100):
                scheduler.step()

        def unpack_data_nyu(data, device="cuda"):
            return [d.to(device) for d in data]

        objective = partial(m_dreg, beta=conf.training.beta)
        t_objective = objective

        best_test_loss = 1e9
        test_worsening_count = 0
        early_stop = False

        def train(epoch):
            model.train()
            b_loss = 0
            loss_series = pd.Series({"lpips": 0, "recon_logprob": 0, "prior": 0, "z_dist": 0})

            if data_parallel:
                train_loader.sampler.set_epoch(epoch)

            pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=rank if data_parallel else 0)

            for i, dataT in pbar:
                data = unpack_data_nyu(dataT, device=device)
                # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                optimizer.zero_grad()

                if use_mixed_precision:
                    with autocast(dtype=torch.float16, device_type="cuda"):
                        loss, loss_dict = objective(model, data,
                                          K=conf.training.K,
                                          data_parallel=data_parallel,
                                          wlpips=conf.training.wlpips,
                                          wadversary=conf.training.wadversary,
                                          train_adversary=None,
                                          lpips=lpips,
                                          channel_weighting=class_weights,
                                          z_close_weight=conf.training.z_close_weight)
                        loss = -loss

                    scaler.scale(loss).backward()
                    if grad_clipper is not None:
                        scaler.unscale_(optimizer)
                        grad_clipper.autoclip_grad()
                    scaler.step(optimizer)


                else:
                    loss, loss_dict = objective(model, data,
                                      K=conf.training.K,
                                      data_parallel=data_parallel,
                                      wlpips=conf.training.wlpips,
                                      wadversary=conf.training.wadversary,
                                      train_adversary=None,
                                      lpips=lpips,
                                      channel_weighting=class_weights,
                                      z_close_weight=conf.training.z_close_weight)

                    loss = -loss

                    loss.backward()
                    # print(f"grad norm: {get_total_norm(model.parameters())}")
                    # grad_norm_sum = (grad_norm_sum + get_total_norm(model.parameters()))/(count + 1)

                    if grad_clipper is not None:
                        grad_clipper.autoclip_grad()
                    optimizer.step()

                loss_series -= pd.Series(loss_dict)

                b_loss += loss.detach().item()
                if conf.print_freq > 0 and i % conf.print_freq == 0:
                    print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))
                    print(f"loss breakdown: {loss_dict}")
                pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))

                if i % scheduler_step_freq == 0:
                    scheduler.step()

                if enable_wandb and i == 0:
                    for m, (name, dbatch) in enumerate(zip(modalities, data)):
                        dbatch = dbatch.detach().cpu()
                        dbatch = torch.stack([display_data_fs[m](d) for d in dbatch], dim=0)
                        grid = make_grid(dbatch, nrow=8)
                        wandb.log({f"Data/{name}": wandb.Image(grid)}, step=epoch)

            # Epoch loss
            epoch_loss = b_loss / len(train_loader.dataset)
            loss_series = loss_series / len(train_loader.dataset)
            if enable_wandb:
                wandb.log({"Loss/train": epoch_loss}, step=epoch)
                wandb.log({f"Loss/train/breakdown/{key}": val for key, val in loss_series.items()}, step=epoch)
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))
            print(loss_series)


        def test(epoch):
            b_loss = 0
            loss_series = pd.Series({"lpips": 0, "recon_logprob": 0, "prior": 0, "z_dist": 0})

            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = unpack_data_nyu(dataT, device=device)
                    loss, loss_dict = t_objective(model, data,
                                        K=conf.training.K,
                                        test=True,
                                        data_parallel=data_parallel,
                                        wlpips=conf.training.wlpips,
                                        wadversary=conf.training.wadversary,
                                        train_adversary=None,
                                        lpips=lpips,
                                        channel_weighting=class_weights,
                                        z_close_weight=conf.training.z_close_weight)

                    loss = -loss

                    b_loss += loss.detach().item()
                    loss_series -= pd.Series(loss_dict)

                    if i == 0 and epoch % 1 == 0:
                        # Compute cross-generations
                        N_gen = 3
                        n_samples = conf.training.batch_size // N_gen
                        limited_data = [d[:n_samples] for d in data]

                        cg_imgs = []
                        for n in range(N_gen):
                            cg_imgs.append(model("self_and_cross_modal_generation", limited_data))

                        cg_imgs = stack_generation_matrices(cg_imgs, limited_data, display_data_fs)

                        for i in range(NUM_VAES):
                            for j in range(NUM_VAES):
                                if enable_wandb:
                                    wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Image(cg_imgs[i][j])}, step=epoch)
                        del cg_imgs

                        # Also save some example latent vectors
                        latents = model.module("encode", data) if data_parallel else model("encode", data)
                        reconstructions = model.module("decode", latents) if data_parallel else model("decode", latents)

                        latent_dim_w = model.module.module.w_channels if data_parallel else model.module.w_channels
                        latent_dim_z = model.module.module.params.z_channels if data_parallel else model.module.z_channels

                        for j in range(NUM_VAES):
                            latent_img = latents[:, j:j + latent_dim_w].detach().cpu()
                            latent_img = torch.transpose(latent_img, 0, 1).reshape(-1, *latent_img.shape[2:]).unsqueeze(1)
                            latent_img = make_grid(latent_img, nrow=conf.training.batch_size)

                            if enable_wandb:
                                wandb.log({'Latents/wm{}'.format(j): wandb.Image(latent_img)}, step=epoch)

                                recon_avg = reconstructions[j].mode if isinstance(reconstructions[i], torch.distributions.Bernoulli) else reconstructions[j].mean
                                recon_display = torch.stack([display_data_fs[j](d) for d in recon_avg.detach().cpu()], dim=0)
                                wandb.log({f'Reconstructions/m{j}': wandb.Image(recon_display)}, step=epoch)

                        latent_img = latents[:, -latent_dim_z:].detach().cpu()
                        latent_img = torch.transpose(latent_img, 0, 1).reshape(-1, *latent_img.shape[2:]).unsqueeze(1)
                        latent_img = make_grid(latent_img, nrow=conf.training.batch_size)

                        if enable_wandb:
                            wandb.log({'Latents/z': wandb.Image(latent_img)}, step=i)


            # Epoch test loss
            epoch_loss = b_loss / len(test_loader.dataset)
            if enable_wandb:
                wandb.log({"Loss/test": epoch_loss}, step=epoch)
                wandb.log({f"Loss/test/breakdown/{key}": val for key, val in loss_series.items()}, step=epoch)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))
            print(loss_series)

            return epoch_loss


        with Timer('MMVAEplus') as t:
            # test(0)

            for epoch in range(conf.training.start_epoch + 1, conf.training.num_epochs + 1 + conf.training.start_epoch):
                train(epoch)
                test_loss = test(epoch)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    test_worsening_count = 0

                if rank == 0 or not data_parallel:
                    save_model_light(model.module if data_parallel else model,
                                    runPath + '/model_' + str(epoch) + '.rar')
                
                else:
                    test_worsening_count += 1
                    if test_worsening_count > conf.training.patience:
                        print(f"Early stopping. Best test loss attained in epoch: {epoch - conf.training.patience}")
                        break

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


@hydra.main(version_base=None, config_path="configs/training/", config_name="celeba_semseg_autoencoder_train.yaml")
def main(conf: DictConfig) -> None:
    conf = expand_sub_confs(conf)

    # Random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    # start the training function, possibly with data parallelism
    if isinstance(conf.device, str):
        run_training(conf.device, conf, False)
    elif isinstance(conf.device, Iterable):
        mp.spawn(
            run_training,
            args=(conf, True, len(conf.device)),
            nprocs=len(conf.device),
            join=True)


if __name__ == '__main__':
    main()

