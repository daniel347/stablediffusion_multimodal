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
from functools import reduce

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

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ray.cloudpickle as pickle
import tempfile

from ldm_util import instantiate_from_config, dict_to_namespace, namespace_to_dict, Timer, save_model_light
from models.dataset_NYU_Depth import NYUDepth
from utils import expand_sub_confs, override_config

import pyiqa

enable_wandb = False

optimise_hyperparameters = False


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
        raise NotImplementedError("Initialising from stable diffusion not implemented yet")

    elif conf.model_initialisation is not None:
        print("Initialising with custom weights")
        model.load_state_dict(torch.load(conf.model_initialisation))

    return model


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
                pass
                # shutil.rmtree(runPath)
            else:
                os.makedirs(runPath, exist_ok=True)
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

        # preparation for training
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=conf.training.lr, amsgrad=True)

        scheduler = ChainedScheduler([LinearLR(optimizer, 0.1, 1, total_iters=10),
                                      ExponentialLR(optimizer, 0.998)])

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

        # Load NYU dataset
        kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
        tx = v2.Compose([v2.Resize([64, 64])])

        train_dataset = NYUDepth(conf.data.datadir, transform=tx, train=True, modality=['colour', 'depth'])
        test_dataset = NYUDepth(conf.data.datadir, transform=tx, train=False, modality=['colour', 'depth'])

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if data_parallel else None
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if data_parallel else None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.training.batch_size, shuffle=not data_parallel, sampler=train_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.training.batch_size, shuffle=False, sampler=test_sampler, **kwargs)

        # Set the learning rate forward to continue training after pause
        if conf.training.start_epoch > 0:
            for _ in range(len(train_loader)//scheduler_step_freq):
                scheduler.step()

        def unpack_data_nyu(data, device="cuda"):
            return [data[0].to(device), data[1].to(device)]

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

                latents = model_first_stage.encode(data)
                loss, loss_dict = model(latents)

                loss.backward()
                optimizer.step()
                b_loss += loss.detach().item()
                if conf.print_freq > 0 and i % conf.print_freq == 0 and enable_pbars:
                    pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))

                if i % scheduler_step_freq == 0:
                    scheduler.step()
                    print(f"Learning rate: {scheduler.get_last_lr()}")

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
            with torch.no_grad():
                for i, dataT in pbar:
                    data = unpack_data_nyu(dataT, device=device)

                    latents = model_first_stage.encode(data)
                    loss, loss_dict = model(latents)

                    b_loss += loss.detach().item()

                    if i == 0  and epoch % 1 == 0:
                        # Perform test generations
                        N_gen = 64
                        samples_zw = model.module.sample(None, batch_size=N_gen) if data_parallel else model.sample(None, batch_size=N_gen)
                        samples_x = model_first_stage.decode(samples_zw)

                        for m in range(NUM_VAES):
                            samples_x_grid = make_grid(samples_x[m].mean, nrow=2)
                            if enable_wandb: wandb.log({f'Generation/m{m}': wandb.Image(samples_x_grid)}, step=epoch)

            # Epoch test loss
            epoch_loss = b_loss / len(test_loader.dataset)
            if enable_wandb: wandb.log({"Loss/test": epoch_loss}, step=epoch)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))

            return epoch_loss

        def test_fid(epoch, n_samples=128, use_diffusion=True):
            gen_batch_size = 64
            n_batches = n_samples // gen_batch_size

            tempdir_colour = os.path.join(runPath, 'temp_fid_colour')
            os.makedirs(tempdir_colour, exist_ok=True)

            tempdir_depth = os.path.join(runPath, 'temp_fid_depth')
            os.makedirs(tempdir_depth, exist_ok=True)

            if enable_pbars:
                pbar = tqdm.tqdm(range(n_batches))
            else:
                pbar = range(n_batches)

            model.eval()
            with torch.no_grad():
                for n in pbar:
                    if use_diffusion:
                        latents = model.module.sample(None, batch_size=gen_batch_size) if data_parallel else model.sample(None, batch_size=gen_batch_size)
                    else:
                        latent_dim = (gen_batch_size, *(model.module.input_shape if data_parallel else model.input_shape))
                        latents = torch.randn(latent_dim, device=device)
                    samples_x = model_first_stage.decode(latents)

                    for i, s in enumerate(samples_x[0].mean[0]):
                        tvu.save_image(s, os.path.join(tempdir_colour, f"sample_colour_{n*gen_batch_size + i}.png"))

                    for i, s in enumerate(samples_x[1].mean[0]):
                        tvu.save_image(s, os.path.join(tempdir_depth, f"sample_depth_{n*gen_batch_size + i}.png"))

            fid = pyiqa.create_metric('fid', device=device)

            colour_fid = fid(tempdir_colour, dataset_name="nyu_colour", dataset_res=299, dataset_split="test")
            depth_fid = fid(tempdir_depth, dataset_name="nyu_depth", dataset_res=299, dataset_split="test")

            if enable_wandb:
                wandb.log({"FID_colour": colour_fid, "FID_depth": depth_fid}, step=epoch)

            print(f'====>             FID colour: {colour_fid:.2f}  depth: {depth_fid:.2f}')

            return colour_fid, depth_fid

        if conf.training.test_only:
            test_epoch(start_epoch)
            fid_colour, fid_depth = test_fid(start_epoch)
            print("FIDs without diffusion")
            test_fid(start_epoch, use_diffusion=False)
            return

        with Timer('MMVAEplus') as t:
            test_epoch(start_epoch)
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
                        train.report({"loss": best_test_loss, "fid_colour": fid_colour, "fid_depth": fid_depth}, checkpoint=checkpoint)

                """
                if epoch % 1 == 0 and (rank == 0 or not data_parallel):
                    save_model_light(model.module if data_parallel else model, runPath + '/model_'+str(epoch)+'.rar')
                """

        print("Finished Training")

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


def run_training_ray_tune(config, conf, **params):
    conf = override_config(conf, config)
    print(conf)

    run_training(conf=conf, **params)


@hydra.main(version_base=None, config_path="../configs", config_name="train_latent_diffusion.yaml")
def main(conf: DictConfig) -> None:
    conf = expand_sub_confs(conf)

    # Random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    if optimise_hyperparameters:
        num_samples = 15
        max_num_epochs = 15
        gpus_per_trial = 1

        output_dir = Path(os.path.join(conf.data.outputdir, conf.experiment, "ray_tune"))
        output_dir.mkdir(parents=True, exist_ok=True)

        assert isinstance(conf.device, str), "Hyperparameter optimisation only works with single GPU"
        config = {
            "model_conf.model.params.unet_config.params.layers_per_level": tune.choice(list(range(1, 4))),
            "model_conf.model.params.unet_config.params.level_channels[0]": tune.choice(list(range(32, 256+16, 16))),
            "model_conf.model.params.unet_config.params.level_channels[1]": tune.choice(list(range(32, 256+16, 16))),
            "model_conf.model.params.unet_config.params.act": tune.choice(["swish", "leaky_relu", "gelu"]),
            "model_conf.model.params.unet_config.params.norm": tune.choice(["group", "batch", "layer"]),
            "training.lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
            # "training.batch_size": tune.choice([8, 16, 32, 64]),
        }
        """
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        """

        algo = TuneBOHB(metric="loss", mode="min")
        algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=2)
        scheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            metric="loss",
            mode="min",
            max_t=1800)

        """
        result = tune.run(
            partial(run_training_ray_tune, rank="cuda:0", conf=conf, data_parallel=False, world_size=1, ray_tune=True),
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
        )
        """
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