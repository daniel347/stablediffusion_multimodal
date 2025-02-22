# Train MMVAEplus on CUB Image-Captions dataset
import os
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
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.profiler import ProfilerActivity, profile, record_function
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler, ExponentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.utils as tvu
from torch.amp import GradScaler, autocast

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ray.cloudpickle as pickle
import tempfile

from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
from ldm.util import instantiate_from_config, dict_to_namespace, namespace_to_dict, Timer, save_model_light, save_model_ema
from mmvae_objectives import m_dreg
from ldm.data import get_dataset
import pyiqa

enable_wandb = True


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
    model = instantiate_from_config(conf["model_conf"]["model"])

    if conf.init_from_stable_diffusion:
        raise NotImplementedError("Initialising from stable diffusion not implemented")

    elif conf.model_initialisation is not None:
        print("Initialising with custom weights")
        model.load_state_dict(torch.load(conf.model_initialisation))

    return model

def stack_generation_matrices(mats, data):
    # Given a list of N_gen cross and self generation matrices, get them into a single matix
    n_modes = len(mats[0])
    n_inputs = data[0].shape[0]

    out_mat = [[None for _ in range(n_modes)] for _ in range(n_modes)]
    for i in range(n_modes):
        for j in range(n_modes):
            img_shape = (3, *mats[0][i][j].shape[3:])
            input_data = data[i].unsqueeze(0).detach().cpu()
            output_data = [m[i][j].detach().cpu() for m in mats]

            if i == 1:
                input_data = input_data.repeat(1, 1, 3, 1, 1)
            if j == 1:
                output_data = [m.repeat(1, 1, 3, 1, 1) for m in output_data]
            out_mat[i][j] = make_grid(torch.cat([input_data] + output_data, dim=0).view(-1, *img_shape),
                                      nrow=n_inputs)

    return out_mat


def run_training(rank, conf, data_parallel=False, world_size=None, ray_tune=False, enable_pbars=True):
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

        model_first_stage = model.module.first_stage_model if data_parallel else model.first_stage_model

        NUM_VAES = len(model_first_stage.vaes)
        print(f"Using {NUM_VAES} vaes")

        # Set experiment name if not set
        if not conf.experiment:
            conf.experiment = model.modelName

        # Set up run path
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        if conf.training.test_only:
            runId += "_test"

        if ray_tune:
            runId += f"_rank{train.get_context().get_world_rank()}"

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

            wandb.alert(title=f"Experiment {conf.experiment} has started", text=f"Run id: {runId}")

        # preparation for training
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=conf.training.lr, amsgrad=True)

        scheduler = ChainedScheduler([LinearLR(optimizer, 0.1, 1, total_iters=10),
                                      ExponentialLR(optimizer, conf.training.gamma)])

        scheduler_step_freq = 25

        start_epoch = None

        if ray_tune:
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    start_epoch = checkpoint_state["epoch"]
                    model.load_state_dict(checkpoint_state["net_state_dict"])
                    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

        if start_epoch is None:
            start_epoch = conf.training.start_epoch

        use_mixed_precision = conf.model_conf.model.params.unet_config.params.use_fp16
        if use_mixed_precision:
            print("Using mixed precision training")
            scaler = GradScaler()


        # Load NYU dataset
        modalities = ['colour', 'depth']
        kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
        tx = v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)])

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

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if data_parallel else None
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if data_parallel else None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.training.batch_size, shuffle=not data_parallel, sampler=train_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.training.batch_size, shuffle=False, sampler=test_sampler, **kwargs)

        # Set the learning rate forward to continue training after pause
        if conf.training.start_epoch > 0:
            for _ in range(len(train_loader)//scheduler_step_freq):
                scheduler.step()

        def unpack_data_nyu(data, device="cuda"):
            return [d.to(device) for d in data]

        best_test_loss = 10000
        test_worseining_count = 0
        early_stop = False

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
                data = unpack_data_nyu(dataT, device=device)
                # print(f"data is shape: {[d.shape for d in data]} on device: {data[0].device}")
                optimizer.zero_grad()

                if use_mixed_precision:
                    with autocast(dtype=torch.float16, device_type="cuda"):
                        latents = model_first_stage.encode(data)
                        loss, loss_dict = model(latents)

                    scaler.scale(loss).backward()
                else:
                    latents = model_first_stage.encode(data)
                    print(f"Latent shape: {latents.shape}")
                    loss, loss_dict = model(latents)
                    loss.backward()

                if torch.isnan(loss).item():
                    if enable_wandb:
                        wandb.alert(title="Loss is NaN", text=f"Halting training at epoch {epoch} and step {i}")
                    raise ValueError("Loss has become NaN, halting training")

                optimizer.step()
                b_loss += loss.detach().item()
                if conf.print_freq > 0 and i % conf.print_freq == 0 and enable_pbars:
                    pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))

                model.on_train_batch_end()  # Updates EMA if present

                if i % scheduler_step_freq == 0:
                    scheduler.step()
                    # print(f"Learning rate: {scheduler.get_last_lr()}")

                if enable_wandb and i == 0:
                    for name, d in zip(modalities, data):
                        grid = make_grid(d, nrow=8)
                        wandb.log({f"Data/{name}": wandb.Image(grid)}, step=epoch)


            # Epoch loss
            epoch_loss = b_loss / len(train_loader.dataset)
            if enable_wandb: wandb.log({"Loss/train": epoch_loss}, step=epoch)
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))

        def test_epoch(epoch):
            if enable_pbars:
                pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0)
            else:
                pbar = enumerate(test_loader)

            b_loss = 0
            with torch.no_grad(), model.ema_scope("Epoch test"):
                for i, dataT in pbar:
                    data = unpack_data_nyu(dataT, device=device)

                    latents = model_first_stage.encode(data)
                    loss, loss_dict = model(latents)

                    b_loss += loss.detach().item()

                    if i == 0 and epoch % 1 == 0:
                        # Perform test generations
                        N_gen = 16
                        samples_zw = model.module.sample(None, batch_size=N_gen) if data_parallel else model.sample(
                            None, batch_size=N_gen)
                        samples_x = model_first_stage.decode(samples_zw)

                        for m in range(NUM_VAES):
                            samples_x_grid = make_grid(samples_x[m].mean, nrow=2)
                            print(samples_x_grid.shape)
                            if enable_wandb: wandb.log({f'Generation/m{m}': wandb.Image(samples_x_grid)}, step=epoch)

            # Epoch test loss
            epoch_loss = b_loss / len(test_loader.dataset)
            if enable_wandb: wandb.log({"Loss/test": epoch_loss}, step=epoch)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))

            return epoch_loss

        def test_fid(epoch, n_samples=32, use_diffusion=True):
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
                        latents = model.module.sample(None,
                                                      batch_size=gen_batch_size) if data_parallel else model.sample(
                            None, batch_size=gen_batch_size)
                    else:
                        latent_dim = (
                        gen_batch_size, *(model.module.input_shape if data_parallel else model.input_shape))
                        latents = torch.randn(latent_dim, device=device)
                    samples_x = model_first_stage.decode(latents)

                    for name, samples in zip(modalities, samples_x):
                        for i, s in enumerate(samples.mean):
                            tvu.save_image(s, os.path.join(tempdirs[name], f"sample_{name}_{n * gen_batch_size + i}.png"))

            fid = pyiqa.create_metric('fid', device=device)

            fid_scores = []
            for mod in modalities:
                split = "test" if conf.data.dataset_name == "nyu_depth" else "val"
                fid_score = fid(tempdirs[mod], dataset_name=f"{conf.data.dataset_name}_{mod}", dataset_res=299, dataset_split=split)
                if enable_wandb:
                    wandb.log({f"FID_{mod}": fid_score}, step=epoch)

                fid_scores.append(fid_score)

                print(f'====>             FID {mod}: {fid_score:.2f}')

            return fid_scores


        with Timer('MMVAEplus') as t:
            test_epoch(start_epoch)
            fid_colour, fid_depth = test_fid(start_epoch)
            print(f"FID colour: {fid_colour}   FID depth: {fid_depth}")
            for epoch in range(start_epoch + 1, conf.training.num_epochs + 1 + start_epoch):
                train_epoch(epoch)

                test_loss = test_epoch(epoch)
                fid_colour, fid_depth = test_fid(epoch)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    test_worsening_count = 0
                    if (rank == 0 or not data_parallel) and not ray_tune:
                        save_model_light(model.module if data_parallel else model,
                                         runPath + '/model_' + str(epoch) + '.rar')
                else:
                    test_worsening_count += 1
                    if test_worsening_count > conf.training.patience and not ray_tune:
                        print(f"Early stopping. Best test loss attained in epoch: {epoch - conf.training.patience}")

                        break

                if ray_tune:
                    checkpoint_data = {
                        "epoch": epoch,
                        "net_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    with tempfile.TemporaryDirectory() as checkpoint_dir:
                        data_path = Path(checkpoint_dir) / "data.pkl"
                        with open(data_path, "wb") as fp:
                            pickle.dump(checkpoint_data, fp)

                        checkpoint = Checkpoint.from_directory(checkpoint_dir)
                        train.report({"loss": best_test_loss, "fid_colour": fid_colour, "fid_depth": fid_depth}, checkpoint=checkpoint)  # "fid_depth": fid_depth

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


@hydra.main(version_base=None, config_path="configs/training/", config_name="imagenet_stable_diffusion_train.yaml")
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
            join=True
        )


if __name__ == '__main__':
    main()

