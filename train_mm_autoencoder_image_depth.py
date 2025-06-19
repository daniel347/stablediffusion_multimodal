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
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path

import torch
from torch import optim
from torchvision.transforms import v2, InterpolationMode
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler, ExponentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
import pandas as pd
from torch.profiler import record_function
import torchvision.utils as tvu
import cmmd_pytorch as cmmd

from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
from ldm.util import instantiate_from_config, Timer, save_model_light, GradAutoClipper
from objectives import get_objective, get_adversarial_loss, train_discriminator
from objectives.mmvae_objectives import m_dreg
from ldm.data import get_dataset

from utils import (save_and_log_latents, save_and_log_img, duplicate_stable_diff_weights, expand_sub_confs,
                   move_data_to_device, save_checkpoint, load_checkpoint, load_image_datasets, create_dataloaders,
                   cat_dists_or_tensor_latents, insert_stable_diff_weights)

import pyiqa

enable_wandb = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def initialise_model(conf):
    first_stage_conf = conf.model.params.first_stage_config
    model = instantiate_from_config(first_stage_conf)

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
            ignore_keys += ["encoder.conv_out", "quant_conv", "decoder.conv_in", "post_quant_conv"] if vae.decoder.z_shape[1] != 4 else []
            sd = copy.deepcopy(first_stage_dict)
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]

            # If we have doubled the number of channels we must go though and double all the initialisation weight shapes
            sd = duplicate_stable_diff_weights(sd, vae, alpha=0.25)

            M, U = vae.load_state_dict(sd, strict=False)
            for m in M:
                print(f"Missing key: {m}")
            for u in U:
                print(f"Unused key: {u}")
        model = MMVAEDataParallelWrapper(model)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Trainable parameter count: {params}")

        return model

    return MMVAEDataParallelWrapper(model)


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


def freeze_encoders(model, data_parallel, freeze=True):
    vaes = model.module.module.vaes if data_parallel else model.module.vaes
    for vae in vaes:
        for param in vae.encoder.parameters():
            param.requires_grad_(not freeze)
        vae.quant_conv.requires_grad_(not freeze)


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

        mod = model.module.module if data_parallel else model.module
        vaes = mod.vaes
        NUM_VAES = len(vaes)

        # Set experiment name if not set
        if not conf.experiment:
            conf.experiment = model.modelName

        # Set up run path
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S")) + "_" + conf.name
        if rank == 0 or not data_parallel:
            experiment_dir = Path(os.path.join(conf.outputdir, conf.experiment, "checkpoints"))
            experiment_dir.mkdir(parents=True, exist_ok=True)
            runPath = experiment_dir / runId
            if runPath.exists():
                shutil.rmtree(str(runPath))
            runPath.mkdir(parents=True, exist_ok=True)
            print('Expt:', runPath)
            print('RunID:', runId)

            # save args to run
            with open('{}/args.json'.format(str(runPath)), 'w') as fp:
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
                notes=conf.notes
            )

            wandb.alert(title="Starting Run", text="Started training stable diffusion autoencoder")
        else:
            run = None

        if "objective_is_z_combiner" in conf.training and conf.training.objective_is_z_combiner:
            print("Training the z combiner network - Freezing parameters of the "
                  "encoders and using the z combiner network")
            freeze_encoders(model, data_parallel, freeze=True)
            mod.use_z_comb_network = True  # Default to using the z combiner when encoding

        # preparation for training
        if "lr_decoder" in conf.training and conf.training.lr_decoder is not None:
            # Use a different learning rate for the decoder, primarily for z combiner training
            optim_args = [{"params": vae.decoder.parameters(), "lr": conf.training.lr_decoder} for vae in vaes]
            optim_args += [{"params": vae.post_quant_conv.parameters(), "lr": conf.training.lr_decoder} for vae in vaes]
            optim_args += [{"params": mod.z_comb_net.parameters(), "lr": conf.training.lr}]
            optimizer = optim.Adam(optim_args, lr=conf.training.lr, amsgrad=True)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=conf.training.lr, amsgrad=True)

        scheduler = ChainedScheduler([LinearLR(optimizer, 0.1, 1, total_iters=10),
                                      ExponentialLR(optimizer, 0.99)])

        use_adversary = ("discriminator" in conf.training and conf.training.discriminator is not None)
        if use_adversary:
            print("Initialising discriminator for adversarial loss ...")
            disc = instantiate_from_config(conf.training.discriminator).to(device)
            disc_optimizer = optim.Adam(disc.parameters(), lr=conf.training.discriminator.discriminator_lr, amsgrad=True)

            get_adversarial_loss_f = partial(get_adversarial_loss, discriminator=disc, device=device)
            train_discriminator_f = partial(train_discriminator, discriminator=disc, optimiser=disc_optimizer, device=device)
        else:
            get_adversarial_loss_f = None
            train_discriminator_f = None
            assert conf.training.objective.params.wadversary == 0.0

        scheduler_step_freq = 50

        start_iter = conf.training.start_iter

        if conf.model_initialisation is not None:
            print("Initialising with custom weights")
            mod = model.module if data_parallel else model
            if "resume_training" in conf and conf.resume_training:
                load_kwargs = {"discriminator": disc,
                               "discriminator_optimizer": disc_optimizer} if use_adversary else {}
                start_iter = load_checkpoint(None, mod, optimizer, scheduler, conf.model_initialisation, **load_kwargs)
            else:
                # just load the model
                load_kwargs = {"discriminator": disc,
                               "discriminator_optimizer": disc_optimizer} if use_adversary else {}
                start_iter = load_checkpoint(None, mod, None, None, conf.model_initialisation, **load_kwargs)

        use_mixed_precision = conf.training.use_mixed_precision
        if use_mixed_precision:
            print("Using mixed precision training")
            scaler = GradScaler()

        # Load dataset
        train_dataset, test_dataset = load_image_datasets(conf.data)
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, conf.training.batch_size, world_size, rank, data_parallel)

        data_conf_0 = conf.data[0] if isinstance(conf.data, ListConfig) else conf.data
        data_root_dirs = [Path(cd.root_data_dir) for cd in conf.data] if isinstance(conf.data, ListConfig) else [
            Path(conf.data.root_data_dir)]
        dataset_name = conf.dataset_name
        modality = data_conf_0.modality

        if isinstance(train_dataset, torch.utils.data.ConcatDataset):
            display_data_fs = [getattr(train_dataset.datasets[0], f"{mode}_display") for mode in data_conf_0.modality]
        else:
            display_data_fs = [getattr(train_dataset, f"{mode}_display") for mode in data_conf_0.modality]


        print(f"data loader lengths train: {len(train_loader)}, test: {len(test_loader)}")

        grad_clipper = None
        if "grad_clipper" in conf.training:
            grad_clipper = GradAutoClipper(model, **conf.training.grad_clipper)

        if start_iter > 0 and start_iter < 50:
            # this is almost certainly an epoch number
            start_iter *= len(train_loader)

        start_epoch = start_iter // len(train_loader)

        objective_opts = OmegaConf.to_container(conf.training.objective.params)
        if "channel_weighting" in objective_opts and objective_opts["channel_weighting"] is not None:
            print("using inverse class frequency weighting for semantic segmentation loss")
            class_freqs = torch.load(os.path.join(data_conf_0.root_data_dir, objective_opts["channel_weighting"]))
            class_weights = 1.0 / class_freqs
            class_weights /= class_weights.sum()
            class_weights = class_weights.to(device)

            class_weights = [class_weights if m == "semseg" else None for m in modality]
            objective_opts["channel_weighting"] = class_weights
        else:
            objective_opts["channel_weighting"] = [None for _ in modality]

        objective_opts["get_adversarial_loss_f"] = get_adversarial_loss_f
        objective_opts["model"] = model
        objective_opts["device"] = device
        objective = get_objective(conf.training.objective.name, **objective_opts)
        t_objective = objective
        
        dataloaders = {"train_dataset": train_dataset,
                       "test_dataset": test_dataset,
                       "train_loader": train_loader,
                       "test_loader": test_loader}

        compute_metrics = conf.training.compute_metrics if "compute_metrics" in conf.training else False

        def train(subepoch, itercount, epochcount, n_iter, train_pbar):
            model.train()
            b_loss = 0
            loss_series = None

            n_stopiter = n_iter + itercount

            if enable_wandb:
                wandb.log({"subepoch": subepoch}, step=itercount)

            while itercount < n_stopiter:
                if data_parallel:
                    train_loader.sampler.set_epoch(epochcount)

                if enable_wandb:
                    wandb.log({"epoch": epochcount}, step=itercount)

                for i, dataT in train_pbar:
                    data = move_data_to_device(dataT, device=device)
                    # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                    optimizer.zero_grad()

                    if use_mixed_precision:
                        with autocast(dtype=torch.float16, device_type="cuda"):
                            loss, loss_dict, model_outs = objective(model, data, K=conf.training.K, iteration=itercount)
                            loss = -loss

                        scaler.scale(loss).backward()
                        if grad_clipper is not None:
                            scaler.unscale_(optimizer)
                            grad_clipper.autoclip_grad()
                        scaler.step(optimizer)


                    else:
                        loss, loss_dict, model_outs = objective(model, data, K=conf.training.K, iteration=itercount)
                        loss = -loss
                        loss.backward()

                        if grad_clipper is not None:
                            grad_clipper.autoclip_grad()
                        optimizer.step()

                    if use_adversary:
                        disc_loss = 0
                        if isinstance(model_outs[1][0], list):
                            for px_u in model_outs[1]:
                                disc_loss = disc_loss + train_discriminator_f(data, px_u)
                        else:
                            disc_loss = train_discriminator_f(data, model_outs[1])
                        loss_dict["discriminator_loss"] = -disc_loss.detach().cpu().sum().item()

                    if loss_series is None:
                        loss_series = -pd.Series(loss_dict)
                    else:
                        loss_series -= pd.Series(loss_dict)

                    b_loss += loss.detach().item()
                    if conf.print_freq > 0 and i % conf.print_freq == 0:
                        print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))
                        print(f"loss breakdown: {loss_dict}")
                    train_pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))

                    if i % scheduler_step_freq == 0:
                        scheduler.step()

                    if enable_wandb:
                        wandb.log({"Loss/train": loss}, step=itercount)
                        wandb.log({f"Loss/train/breakdown/{key}": -val for key, val in loss_dict.items()}, step=itercount)
                        if i == 0:
                            for m, (name, dbatch) in enumerate(zip(modality, data)):
                                dbatch = dbatch.detach().cpu()
                                dbatch = torch.stack([display_data_fs[m](d) for d in dbatch], dim=0)
                                grid = make_grid(dbatch, nrow=8)
                                wandb.log({f"Data/{name}": wandb.Image(grid)}, step=itercount)

                    itercount += 1
                    if itercount >= n_stopiter:
                        break

                if itercount < n_stopiter:
                    print("Finished one epoch")
                    train_pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                                           position=rank if data_parallel else 0)
                    epochcount += 1

            # Epoch loss
            subepoch_loss = b_loss / n_iter
            loss_series = loss_series / n_iter
            """
            if enable_wandb:
                wandb.log({"Loss/train": epoch_loss}, step=step)
                wandb.log({f"Loss/train/breakdown/{key}": val for key, val in loss_series.items()}, step=step)
            """
            print('====> Subepoch: {:03d} Train loss: {:.4f}'.format(subepoch, subepoch_loss))
            print(loss_series)
            return epochcount, train_pbar

        def test(subepoch, itercount, epochcount, compute_metrics=False, metrics=["fid", "kid"]):
            b_loss = 0
            loss_series = None

            vaes = model.module.module.vaes if data_parallel else model.module.vaes
            mod = model.module.module if data_parallel else model.module

            if enable_wandb:
                wandb.log({"epoch": epochcount, "subepoch": subepoch}, step=itercount)

            if compute_metrics:
                tempdirs = {}
                for mode in modality:
                    tempdir = runPath / f'temp_fid_{mode}'
                    tempdir.mkdir(parents=True, exist_ok=True)
                    tempdirs[mode] = tempdir

            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = move_data_to_device(dataT, device=device)
                    loss, loss_dict, model_outs = t_objective(model, data, K=conf.training.K, test=True, iteration=itercount)
                    loss = -loss

                    if use_adversary:
                        disc_loss = 0
                        if isinstance(model_outs[1][0], list):
                            for px_u in model_outs[1]:
                                disc_loss = disc_loss + train_discriminator_f(data, px_u, test=True)
                        else:
                            disc_loss = train_discriminator_f(data, model_outs[1], test=True)
                        loss_dict["discriminator_loss"] = -disc_loss.detach().cpu().sum().item()

                    b_loss += loss.detach().item()

                    if loss_series is None:
                        loss_series = -pd.Series(loss_dict)
                    else:
                        loss_series -= pd.Series(loss_dict)

                    if compute_metrics:
                        for m, name in enumerate(modality):
                            out = model_outs[1][m][m] if isinstance(model_outs[1][0], list) else model_outs[1][m]
                            for j, img in enumerate(out.mean[0]):
                                s_disp = display_data_fs[m](img.detach().cpu())
                                tvu.save_image(s_disp, str(tempdirs[name] / f"sample_{name}_{i * conf.training.batch_size + j}.png"))

                    if i == 0:  # select a different scene
                        # Compute cross-generations, unless we are using z combiner in which case it will be computed below
                        if mod.can_cross_generate and not mod.use_z_comb_network:
                            N_gen = 1
                            n_samples = conf.training.batch_size // N_gen
                            limited_data = [d[:n_samples] for d in data]

                            cg_imgs = []
                            for n in range(N_gen):
                                cg_imgs.append(model("self_and_cross_modal_generation", limited_data, all_combinations=False))

                            cg_imgs = stack_generation_matrices(cg_imgs, limited_data, display_data_fs)

                            for i in range(NUM_VAES):
                                for j in range(NUM_VAES):
                                    if enable_wandb:
                                        wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Image(cg_imgs[i][j])}, step=itercount)
                            del cg_imgs

                        # Also save some example latent vectors and example reconstructions (without averaging z)
                        latents = model.module("encode", data, avg_z=False) if data_parallel else model("encode", data, avg_z=False)

                        if mod.factorised_latent_space:
                            num_zs = len(latents) - NUM_VAES
                            for j in range(num_zs):  # Choice of z
                                latent_w_img = latents[j]
                                latent_z_img = latents[j + NUM_VAES]

                                save_and_log_latents(latent_w_img, f'Latents/wm{j}', runPath, itercount, run)
                                save_and_log_latents(latent_z_img, f'Latents/zm{j}', runPath, itercount, run)

                                # Reconstructions with known w
                                ws = latents[:NUM_VAES]
                                u = torch.cat([*ws, latent_z_img], dim=1)
                                reconstructions = model.module("decode", u) if data_parallel else model("decode", u)

                                for k in range(NUM_VAES):
                                    save_and_log_img(reconstructions[k], f'Reconstructions/z{j}/m{k}',
                                                     runPath, display_data_fs[k], itercount, run)

                                # Reconstructions with random w
                                W_random = []
                                for k in range(NUM_VAES):
                                    pw = vaes[k].pw(*model.module.pw_params)
                                    latents_w = pw.rsample([latent_z_img.shape[0]]).squeeze(1).to(device)
                                    W_random.append(latents_w)
                                latents_random_w = torch.cat([*W_random, latent_z_img], dim=1)
                                recon_random_w = model.module("decode", latents_random_w) if data_parallel else model(
                                    "decode", latents_random_w)

                                for k in range(NUM_VAES):
                                    save_and_log_img(recon_random_w[k], f'ReconRandomW/z{j}/m{k}',
                                                     runPath, display_data_fs[k], itercount, run)
                        else:
                            reconstructions = model.module("decode", latents) if data_parallel else model("decode", latents)

                            for k, rec in enumerate(reconstructions):
                                save_and_log_img(rec, f'Reconstructions/m{k}',
                                                 runPath, display_data_fs[k], itercount, run)

                        if use_adversary:
                            # Log discriminator outputs
                            real_disc_outs = torch.cat(disc(data), dim=0)
                            save_and_log_img(real_disc_outs.detach().cpu(), f"Discriminator/real", runPath, None, itercount,
                                             run)
                            if isinstance(model_outs[1][0], list):
                                for m, px_u in enumerate(model_outs[1]):
                                    fake_disc_out = torch.cat(disc(px_u), dim=0)
                                    save_and_log_img(fake_disc_out.detach().cpu(), f"Discriminator/fake_m{m}", runPath, None, itercount, run)
                            else:
                                fake_disc_out = torch.cat(disc(model_outs[1]), dim=0)
                                save_and_log_img(fake_disc_out.detach().cpu(), f"Discriminator/fake", runPath, None,
                                                 itercount,
                                                 run)

            if compute_metrics:
                metric_scores = {}
                for met in metrics:
                    if met == "cmmd":
                        metric = partial(cmmd.compute_cmmd, device=device)
                    else:
                        metric = pyiqa.create_metric(met, device=device)

                    scores = []
                    for mod in modality:
                        if mod in ["colour", "depth"]:
                            split = "test" if dataset_name == "nyu_depth" else "val"
                            score = metric(tempdirs[mod], dataset_name=f"{dataset_name}_{mod}", dataset_res=299, dataset_split=split, center_crop=True)
                            if enable_wandb:
                                wandb.log({f"{met}_{mod}": score, "epoch": epochcount, "subepoch": subepoch}, step=itercount)

                            scores.append(score)

                            print(f'====> {met} {mod}: {score:.2f}')
                        else:
                            scores.append(None)

                    metric_scores[met] = scores

                for met, score in metric_scores.items():
                    wandb.log({f"{met}_sum": sum(filter(None, score))}, step=itercount)

            # Epoch test loss
            epoch_loss = b_loss / len(test_loader)
            if enable_wandb:
                wandb.log({"Loss/test": epoch_loss}, step=itercount)
                wandb.log({f"Loss/test/breakdown/{key}": val/len(test_loader) for key, val in loss_series.items()}, step=itercount)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))
            print(loss_series)

            return epoch_loss


        def run_grad_check(model, **kwargs):
            x = (torch.randn(1, 3, 32, 32, dtype=torch.float32, requires_grad=True, device=device),
                 (torch.randn(1, 18, 32,32, dtype=torch.float32, requires_grad=True, device=device) > 0.0).float())

            model.train()

            def obj_grad_check(t1, t2):
                loss, loss_dict, model_outs = t_objective(model, [t1, t2], K=conf.training.K, test=False)
                return loss

            grad_check = torch.autograd.gradcheck(obj_grad_check, x, eps=1e-6, atol=1e-4, fast_mode=True)
            print(f"Grad check result: {grad_check}")


        def check_z_closeness(n_samples=10):
            with torch.no_grad():

                z_mses = []
                z_mags = [[] for _ in range(NUM_VAES)]
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = move_data_to_device(dataT, device=device)

                    latents = model.module("encode", data, avg_z=False) if data_parallel else model("encode", data, avg_z=False)
                    latents_z = latents[NUM_VAES:]

                    if enable_wandb:
                        for m, lat_z in enumerate(latents_z):
                            wandb.log({f'Latents/z_enc_{m}': wandb.Image(lat_z)}, step=i)

                    z_mse = torch.linalg.vector_norm(latents_z[0] - latents_z[1], dim=(1, 2, 3))
                    z_mses.append(z_mse.detach().cpu())

                    for j in range(NUM_VAES):
                        z_mags[j].append(torch.linalg.vector_norm(latents_z[j], dim=(1,2,3)).detach().cpu())

                    if (i + 1) * conf.training.batch_size >= n_samples:
                        break

                z_mses = torch.cat(z_mses)
                z_mags = [torch.cat(m) for m in z_mags]

                data = pd.DataFrame({"z_mse": z_mses, **{f"z{j}_mag": z_mags[j] for j in range(NUM_VAES)}})

                plot = sns.histplot(data=data, kde=True)
                plot_save_path = os.path.join(runPath, f"z_mse_dists.png")
                plt.savefig(plot_save_path, dpi=300)

                if enable_wandb:
                    wandb.log({'z_mse': wandb.Histogram(z_mses, num_bins=20)}, step=i+1)


        def check_learned_prior_params(epoch, **kwargs):
            vaes = model.module.module.vaes if data_parallel else model.module.vaes

            def save_log_print_params(params, name):
                mean = params[0].detach().cpu().transpose(0, 1)
                var = params[1].detach().cpu().transpose(0, 1)

                print(f"{name} variance {var.mean()}, min: {var.min()}, max: {var.max()}")
                save_and_log_latents(mean, name+"_mean", runPath, epoch, run)
                save_and_log_latents(var, name+"_std", runPath, epoch, run)

            for m, vae in enumerate(vaes):
                save_log_print_params(vae.pw_params_aux, f"prior/pw_aux{m}")

            save_log_print_params(model.module.pw_params, f"prior/pw_global")
            save_log_print_params(model.module.pz_params, f"prior/pz_global")


        def decode_encode_latent():
            # Note this is a simple test designed for unimodal data (ie StableIndigo work)
            # - I should make a different file but im a little lazy

            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                                          position=rank if data_parallel else 0):
                    data = move_data_to_device(dataT, device=device)

                    mod = model.module if data_parallel else model

                    latents = mod("encode", data, avg_z=True)

                    latents_perturbed = latents + (0.05 * torch.std(latents)) * torch.randn_like(latents)

                    reconstructions = mod("decode", latents)
                    reconstructions_perturbed = mod("decode", latents_perturbed)

                    reencoded_latents = mod("encode", [reconstructions_perturbed[0].mean], avg_z=True)
                    reencoded_latents_perturbed = mod("encode", [reconstructions_perturbed[0].mean], avg_z=True)

                    save_and_log_latents(latents.detach().cpu(), f'Latents/original', runPath, i, run)
                    save_and_log_latents(latents_perturbed.detach().cpu(), f'Latents/perturbed', runPath, i, run)
                    save_and_log_latents(reencoded_latents.detach().cpu(), f'Latents/reencoded', runPath, i, run)
                    save_and_log_latents(reencoded_latents_perturbed.detach().cpu(), f'Latents/perturbed_reencoded', runPath, i, run)

                    save_and_log_img(data[0].detach().cpu(), f'Data/input', runPath, display_data_fs[0], i, run)
                    save_and_log_img(reconstructions[0], f'Reconstructions/non_perturbed', runPath, display_data_fs[0], i, run)
                    save_and_log_img(reconstructions_perturbed[0], f'Reconstructions/perturbed', runPath, display_data_fs[0], i, run)

                    if i > 5:
                        break

        def profile_train():
            from torch.profiler import profile, record_function, ProfilerActivity

            train_pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                                   position=rank if data_parallel else 0)

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
                train(0, 0, 0, 5, train_pbar)

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            prof.export_chrome_trace(str(runPath / "train_trace.json"))


        def check_decoder_scaling():
            """We do a test to see if the decoder is more sensitive to low-std channels or if the low-std channels are ignored"""

            itercount = 0

            vaes = model.module.module.vaes if data_parallel else model.module.vaes
            mod = model.module if data_parallel else model
            mod.module.use_z_comb_network = True  # Default to using the z combiner when encoding

            with ((torch.no_grad())):
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                                          position=rank if data_parallel else 0):
                    data = move_data_to_device(dataT, device=device)
                    # print(f"data shapes: {data[0].shape}, {data[1].shape}")

                    for m, (name, dbatch) in enumerate(zip(modality, data)):
                        dbatch = dbatch.detach().cpu()
                        dbatch = torch.stack([display_data_fs[m](d) for d in dbatch], dim=0)
                        grid = make_grid(dbatch, nrow=8)
                        wandb.log({f"Data/{name}": wandb.Image(grid)}, step=itercount)

                    latents = mod("encode", data)
                    if isinstance(latents, list):
                        latents = cat_dists_or_tensor_latents(latents)
                        print(latents.shape)

                    for c in range(latents.shape[1]):
                        print(f"Normalising channel {c}")
                        channel_std = torch.ones(latents.shape[1]).to(device)
                        channel_mean = torch.zeros(latents.shape[1]).to(device)

                        channel_std[c] = torch.std(latents[:, c, ...], dim=(0, 1, 2))
                        channel_mean[c] = torch.mean(latents[:, c, ...], dim=(0, 1, 2))
                        latent_norm = ((latents - channel_mean[None, :, None, None]) / channel_std[None, :, None, None]) + channel_mean[None, :, None, None]
                        wandb.log({"channel": c,
                                   "orig_channel_std": channel_std[c].detach().cpu(),
                                   "orig_channel_mean": channel_mean[c].detach().cpu()}, step=itercount)

                        reconstructions = mod("decode", latent_norm)

                        for k, rec in enumerate(reconstructions):
                            save_and_log_img(rec, f'Reconstructions/m{k}',
                                             runPath, display_data_fs[k], itercount, run)

                        itercount += 1

                    # add a fixed amount of noise to each channel and measure the reconstruction error
                    baseline_recon = mod("decode", latents)
                    baseline_error = [torch.nn.functional.mse_loss(d, b.mean) for d, b in zip(data, baseline_recon)]
                    noise = torch.randn_like(latents)

                    wandb.log({f"noiseless_recon_error_mode{m}": baseline_error[m] for m in range(len(modality))}, step=itercount)

                    for c in range(latents.shape[1]):
                        latent_noised = copy.deepcopy(latents)
                        latent_noised[:, c] += 0.2 * noise[:, 0]  # we add the same noise to all channels incase there is a "black hole"

                        recons = mod("decode", latent_noised)
                        error = [torch.nn.functional.mse_loss(d, r.mean) for d, r in zip(data, recons)]

                        wandb.log({"channel": c, **{f"recon_error_mode{m}": error[m].detach().cpu() for m in range(len(modality))}}, step=itercount)

                        for k, rec in enumerate(recons):
                            save_and_log_img(rec, f'NoisyReconstructions/m{k}',
                                             runPath, display_data_fs[k], itercount, run)

                        itercount += 1

                    break

        def training_loop():
            best_test_loss = 1e20
            test_worsening_count = 0

            epochcount = 0

            n_iter = conf.training.test_every_iter if "test_every_iter" in conf.training else len(train_loader)

            train_pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader),
                                   position=rank if data_parallel else 0)

            with Timer('MMVAEplus') as t:
                # test(0, conf.training.start_iter, 0)

                for subepoch, itercount in enumerate(range(start_iter,
                                                           conf.training.num_epochs * n_iter + start_iter,
                                                           n_iter)):
                    epochcount, train_pbar = train(subepoch, itercount, epochcount, n_iter, train_pbar)
                    test_loss = test(subepoch, itercount + n_iter, epochcount, compute_metrics)

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        test_worsening_count = 0
                    else:
                        test_worsening_count += 1
                        if test_worsening_count > conf.training.patience:
                            print(f"Early stopping. Best test loss attained in subepoch: {subepoch - conf.training.patience}")
                            break

                    if rank == 0 or not data_parallel:
                        mod = model.module if data_parallel else model
                        save_kwargs = {"discriminator": disc, "discriminator_optimizer": disc_optimizer} if use_adversary else {}
                        save_checkpoint(None, mod, optimizer, scheduler, itercount, runPath / f"model_{itercount}.pt", **save_kwargs)


        if conf.function.name is not None:
            locals()[conf.function.name](**conf.function.params)
        else:
            training_loop()

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


@hydra.main(version_base=None, config_path="configs/", config_name="celeba_semseg_autoencoder_train")
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

