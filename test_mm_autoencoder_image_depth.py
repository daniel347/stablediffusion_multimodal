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

from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
from ldm.util import instantiate_from_config, dict_to_namespace, namespace_to_dict, Timer, save_model_light
from mmvae_objectives import m_dreg
from ldm.data.dataset_NYU_Depth import NYUDepth

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
            M, U = vae.load_state_dict(copy.deepcopy(first_stage_dict), strict=False)
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


def run_testing(rank, conf, data_parallel=False, world_size=None):
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
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S")) + "_test"
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

        # Load NYU dataset
        kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
        tx = v2.Compose([v2.Resize([256, 256])])

        test_dataset = NYUDepth(conf.data.datadir, transform=tx, train=False, modality=['colour', 'depth'])
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if data_parallel else None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, sampler=test_sampler, **kwargs)

        def unpack_data_nyu(data, device="cuda"):
            return [data[0].to(device), data[1].to(device)]

        objective = partial(m_dreg, beta=conf.training.beta)
        t_objective = objective

        def test():
            epoch = 0
            b_loss = 0
            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = unpack_data_nyu(dataT, device=device)
                    loss = -t_objective(model, data, K=conf.training.K, test=True, data_parallel=data_parallel)
                    b_loss += loss.detach().item()
                    if i == 0  and epoch % 1 == 0:
                        # Compute cross-generations
                        N_gen = 3
                        n_samples = 2
                        limited_data = [d[:n_samples] for d in data]

                        cg_imgs = []
                        for n in range(N_gen):
                            cg_imgs.append(model("self_and_cross_modal_generation", limited_data))

                        cg_imgs = stack_generation_matrices(cg_imgs, limited_data)

                        for i in range(NUM_VAES):
                            for j in range(NUM_VAES):
                                if enable_wandb:
                                    wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Image(cg_imgs[i][j])}, step=epoch)
                        del cg_imgs

            # Epoch test loss
            epoch_loss = b_loss / len(test_loader.dataset)
            if enable_wandb:
                wandb.log({"Loss/test": epoch_loss}, step=epoch)
            print('====>             Test loss: {:.4f}'.format(epoch_loss))


        def visualise_latents(n_samples):
            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = unpack_data_nyu(dataT, device=device)

                    tvu.save_image(data[0], os.path.join(runPath, f"batch_{i}_colour.png"), nrow=conf.training.batch_size)
                    tvu.save_image(data[1], os.path.join(runPath, f"batch_{i}_depth.png"), nrow=conf.training.batch_size)

                    latents = model.module("encode", data) if data_parallel else model("encode", data)
                    print(latents.shape)
                    reconstructions = model.module("decode", latents) if data_parallel else model("decode", latents)

                    latent_dim_w = model.module.module.w_channels if data_parallel else model.module.w_channels
                    latent_dim_z = model.module.module.params.z_channels if data_parallel else model.module.z_channels

                    for j in range(NUM_VAES):
                        latent_img = latents[:, j:j+latent_dim_w].detach().cpu()
                        latent_img = torch.transpose(latent_img, 0, 1).reshape(-1, *latent_img.shape[2:]).unsqueeze(1)
                        latent_img = make_grid(latent_img, nrow=conf.training.batch_size)

                        latent_img_path = os.path.join(runPath, f"batch_{i}_latents_wm{j}.png")
                        tvu.save_image(latent_img, latent_img_path, normalize=True)

                        recon_img_path = os.path.join(runPath, f"batch_{i}_recon_m{j}.png")
                        tvu.save_image(reconstructions[j].mean, recon_img_path)

                        if enable_wandb:
                            wandb.log({'Latents/wm{}'.format(j): wandb.Image(latent_img)}, step=i)
                            wandb.log({f'Reconstructions/m{j}': wandb.Image(reconstructions[j].mean)}, step=i)

                    latent_img = latents[:, -latent_dim_z:].detach().cpu()
                    latent_img = torch.transpose(latent_img, 0, 1).reshape(-1, *latent_img.shape[2:]).unsqueeze(1)
                    latent_img = make_grid(latent_img, nrow=conf.training.batch_size)

                    latent_img_path = os.path.join(runPath, f"batch_{i}_latents_z.png")
                    tvu.save_image(latent_img, latent_img_path, normalize=True)

                    if enable_wandb:
                        wandb.log({'Latents/z': wandb.Image(latent_img)}, step=i)

                    if (i + 1) * conf.training.batch_size >= n_samples:
                        break


        with Timer('MMVAEplus') as t:
            visualise_latents(10)

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


@hydra.main(version_base=None, config_path="configs/testing/", config_name="nyu_depth_autoencoder_test.yaml")
def main(conf: DictConfig) -> None:
    conf = expand_sub_confs(conf)

    # Random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    # start the training function, possibly with data parallelism
    if isinstance(conf.device, str):
        run_testing(conf.device, conf, False)
    elif isinstance(conf.device, Iterable):
        mp.spawn(
            run_testing,
            args=(conf, True, len(conf.device)),
            nprocs=len(conf.device),
            join=True
        )


if __name__ == '__main__':
    main()

