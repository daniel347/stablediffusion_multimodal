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
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

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

from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
from ldm.util import instantiate_from_config, Timer, save_model_light, GradAutoClipper
from objectives import get_objective, train_adversary
from objectives.mmvae_objectives import m_dreg
from ldm.data import get_dataset

from utils import save_and_log_latents, save_and_log_img, duplicate_stable_diff_weights, expand_sub_confs, move_data_to_device

import pyiqa

enable_wandb = True
log_iterations = True

notes = ("We train an autoencoder on NYU depth using a small subset of the new NYUv2 datset, with the new autoencoder loss and architecture.")
name = "new_ae_arch_test"


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.cuda.set_device(rank)
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def initialise_model(conf):
    first_stage_conf = conf.model_conf.model.params.first_stage_config
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
            sd = duplicate_stable_diff_weights(sd, vae)

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

    elif conf.model_initialisation is not None:
        print("Initialising with custom weights")
        model = MMVAEDataParallelWrapper(model)
        model.load_state_dict(torch.load(conf.model_initialisation), strict=False)
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
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S")) + "_" + name
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
            run = wandb.init(
                # Set the project where this run will be logged
                project=conf.experiment,
                # Track hyperparameters and run metadata
                config=OmegaConf.to_container(conf),
                # Run name
                name=f"{runId}_{str(rank)}",
                group=f"DDP_{runId}",
                notes=notes
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

        if "discriminator" in conf.training and conf.training.discriminator is not None:
            disc = instantiate_from_config(conf.training.discriminator)
            disc_optimizer = optim.Adam(disc.parameters(), lr=conf.training.discriminator_lr, amsgrad=True)

            train_adversary_f = partial(train_adversary, discriminator=disc, optimiser=disc_optimizer, device=device)
        else:
            train_adversary_f = None
            assert conf.training.objective.params.wadversary == 0.0

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
                                    v2.CenterCrop(conf.data.input_size)]),
              "depth": v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)])}

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
                                   split="test",
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


        grad_clipper = None
        if "grad_clipper" in conf.training:
            grad_clipper = GradAutoClipper(model, **conf.training.grad_clipper)

        # Set the learning rate forward to continue training after pause
        if conf.training.start_epoch > 0:
            for _ in range(len(train_loader)//scheduler_step_freq):
                scheduler.step()

        objective_opts = OmegaConf.to_container(conf.training.objective.params)
        if "channel_weighting" in objective_opts and objective_opts["channel_weighting"] is not None:
            print("using inverse class frequency weighting for semantic segmentation loss")
            class_freqs = torch.load(os.path.join(conf.data.datadir, objective_opts["channel_weighting"]))
            class_weights = 1.0 / class_freqs
            class_weights /= class_weights.sum()
            class_weights = class_weights.to(device)

            class_weights = [class_weights if m == "semseg" else None for m in modalities]
            objective_opts["channel_weighting"] = class_weights
        else:
            objective_opts["channel_weighting"] = [None for _ in modalities]

        objective_opts["model"] = model
        objective_opts["device"] = device

        objective = get_objective(conf.training.objective.name, **objective_opts)

        t_objective = objective

        def train(epoch):
            model.train()
            b_loss = 0
            loss_series = None

            if data_parallel:
                train_loader.sampler.set_epoch(epoch)

            pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=rank if data_parallel else 0)

            for i, dataT in pbar:
                data = move_data_to_device(dataT, device=device)
                # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                optimizer.zero_grad()

                if use_mixed_precision:
                    with autocast(dtype=torch.float16, device_type="cuda"):
                        loss, loss_dict = objective(model, data, K=conf.training.K)
                        loss = -loss

                    scaler.scale(loss).backward()
                    if grad_clipper is not None:
                        scaler.unscale_(optimizer)
                        grad_clipper.autoclip_grad()
                    scaler.step(optimizer)


                else:
                    loss, loss_dict = objective(model, data, K=conf.training.K)
                    loss = -loss
                    loss.backward()

                    if grad_clipper is not None:
                        grad_clipper.autoclip_grad()
                    optimizer.step()

                if loss_series is None:
                    loss_series = -pd.Series(loss_dict)
                else:
                    loss_series -= pd.Series(loss_dict)

                b_loss += loss.detach().item()
                if conf.print_freq > 0 and i % conf.print_freq == 0:
                    print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))
                    print(f"loss breakdown: {loss_dict}")
                pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))

                if i % scheduler_step_freq == 0:
                    scheduler.step()

                step = epoch * len(train_loader) + i if log_iterations else epoch

                if enable_wandb and i == 0:
                    for m, (name, dbatch) in enumerate(zip(modalities, data)):
                        dbatch = dbatch.detach().cpu()
                        dbatch = torch.stack([display_data_fs[m](d) for d in dbatch], dim=0)
                        grid = make_grid(dbatch, nrow=8)
                        wandb.log({f"Data/{name}": wandb.Image(grid)}, step=step)

            # Epoch loss
            epoch_loss = b_loss / len(train_loader.dataset)
            loss_series = loss_series / len(train_loader.dataset)
            if enable_wandb:
                wandb.log({"Loss/train": epoch_loss}, step=step)
                wandb.log({f"Loss/train/breakdown/{key}": val for key, val in loss_series.items()}, step=step)
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))
            print(loss_series)

        def test(epoch):
            step = (epoch + 1) * len(train_loader) if log_iterations else epoch
            b_loss = 0
            loss_series = None

            vaes = model.module.module.vaes if data_parallel else model.module.vaes
            mod = model.module.module if data_parallel else model.module

            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = move_data_to_device(dataT, device=device)
                    loss, loss_dict = t_objective(model, data, K=conf.training.K, test=True)

                    loss = -loss

                    b_loss += loss.detach().item()

                    if loss_series is None:
                        loss_series = -pd.Series(loss_dict)
                    else:
                        loss_series -= pd.Series(loss_dict)

                    if i == 0 and epoch % 1 == 0:
                        # Compute cross-generations
                        if mod.can_cross_generate:
                            N_gen = 3
                            n_samples = conf.training.batch_size // N_gen
                            limited_data = [d[:n_samples] for d in data]

                            cg_imgs = []
                            for n in range(N_gen):
                                cg_imgs.append(model("self_and_cross_modal_generation", limited_data, all_combinations=False))

                            cg_imgs = stack_generation_matrices(cg_imgs, limited_data, display_data_fs)

                            for i in range(NUM_VAES):
                                for j in range(NUM_VAES):
                                    if enable_wandb:
                                        wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Image(cg_imgs[i][j])}, step=step)
                            del cg_imgs

                        # Also save some example latent vectors and example reconstructions (without averaging z)
                        latents = model.module("encode", data, avg_z=False) if data_parallel else model("encode", data, avg_z=False)

                        if mod.factorised_latent_space:
                            num_zs = len(latents) - NUM_VAES
                            for j in range(num_zs):  # Choice of z
                                latent_w_img = latents[j]
                                latent_z_img = latents[j + NUM_VAES]

                                save_and_log_latents(latent_w_img, f'Latents/wm{j}', runPath, step, run)
                                save_and_log_latents(latent_z_img, f'Latents/zm{j}', runPath, step, run)

                                # Reconstructions with known w
                                ws = latents[:NUM_VAES]
                                u = torch.cat([*ws, latent_z_img], dim=1)
                                reconstructions = model.module("decode", u) if data_parallel else model("decode", u)

                                for k in range(NUM_VAES):
                                    save_and_log_img(reconstructions[k], f'Reconstructions/z{j}/m{k}',
                                                     runPath, display_data_fs[k], step, run)

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
                                                     runPath, display_data_fs[k], step, run)
                        else:
                            reconstructions = model.module("decode", latents) if data_parallel else model("decode", latents)

                            for k, rec in enumerate(reconstructions):
                                save_and_log_img(rec, f'Reconstructions/m{k}',
                                                 runPath, display_data_fs[k], step, run)

            # Epoch test loss
            epoch_loss = b_loss / len(test_loader.dataset)
            if enable_wandb:
                wandb.log({"Loss/test": epoch_loss}, step=step)
                wandb.log({f"Loss/test/breakdown/{key}": val for key, val in loss_series.items()}, step=step)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))
            print(loss_series)

            return epoch_loss


        def run_grad_check(model):
            x = (torch.randn(1, 3, 32, 32, dtype=torch.float32, requires_grad=True, device=device),
                 (torch.randn(1, 18, 32,32, dtype=torch.float32, requires_grad=True, device=device) > 0.0).float())

            model.train()

            def obj_grad_check(t1, t2):
                loss, loss_dict = t_objective(model, [t1, t2], K=conf.training.K, test=False)
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


        def check_learned_prior_params(epoch):
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

        def training_loop():
            best_test_loss = 1e20
            test_worsening_count = 0

            with Timer('MMVAEplus') as t:
                test(-1)

                for epoch in range(conf.training.start_epoch + 1, conf.training.num_epochs + 1 + conf.training.start_epoch):
                    train(epoch)
                    test_loss = test(epoch)

                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        test_worsening_count = 0
                    else:
                        test_worsening_count += 1
                        if test_worsening_count > conf.training.patience:
                            print(f"Early stopping. Best test loss attained in epoch: {epoch - conf.training.patience}")
                            break

                    if rank == 0 or not data_parallel:
                        save_model_light(model.module if data_parallel else model,
                                        runPath + '/model_' + str(epoch) + '.rar')


        if conf.function.name is not None:
            locals()[conf.function.name](**conf.function.params)
        else:
            training_loop()

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


@hydra.main(version_base=None, config_path="configs/training/", config_name="nyu_depth_autoencoder_train.yaml")
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

