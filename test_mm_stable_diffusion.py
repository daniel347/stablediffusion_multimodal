# Train MMVAEplus on CUB Image-Captions dataset
import os
import shutil
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
import tqdm
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import wandb

import torch
from torchvision.transforms import v2
from torchvision.utils import make_grid
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.utils as tvu

from ray import train

from ldm.util import instantiate_from_config, Timer
from ldm.data.dataset_NYU_Depth import NYUDepth
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


def run_testing(rank, conf, data_parallel=False, world_size=None, ray_tune=False, enable_pbars=True):
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
        runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S")) + "_test"

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

        scheduler_step_freq = 25


        # Load NYU dataset
        modalities = ['colour']
        kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
        tx = v2.Compose([v2.Resize([conf.data.input_size, conf.data.input_size])])

        test_dataset = NYUDepth(conf.data.datadir, transform=tx, train=False, modality=modalities)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if data_parallel else None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.training.batch_size, shuffle=False, sampler=test_sampler, **kwargs)

        def unpack_data_nyu(data, device="cuda"):
            return [d.to(device) for d in data]

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

                    if i == 0 and epoch % 1 == 0:
                        # Perform test generations
                        N_gen = 8
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


        def visualise_generated_latents(n_samples=32):
            gen_batch_size = 3
            n_batches = n_samples // gen_batch_size

            if enable_pbars:
                pbar = tqdm.tqdm(range(n_batches))
            else:
                pbar = range(n_batches)

            latent_dim_w = model_first_stage.latent_channel_z
            latent_dim_z = model_first_stage.latent_channel_w

            model.eval()
            with torch.no_grad():
                for n in pbar:
                    if data_parallel:
                        latents = model.module.sample(None, batch_size=gen_batch_size)
                    else:
                        latents = model.sample(None, batch_size=gen_batch_size)

                    reconstructions = model_first_stage.decode(latents)

                    for j in range(NUM_VAES):
                        latent_img = latents[:, j:j+latent_dim_w].detach().cpu()
                        latent_img = torch.transpose(latent_img, 0, 1).reshape(-1, *latent_img.shape[2:]).unsqueeze(1)
                        latent_img = make_grid(latent_img, nrow=gen_batch_size)

                        latent_img_path = os.path.join(runPath, f"batch_{n}_latents_wm{j}.png")
                        tvu.save_image(latent_img, latent_img_path, normalize=True)

                        recon_img_path = os.path.join(runPath, f"batch_{n}_recon_m{j}.png")
                        tvu.save_image(reconstructions[j].mean, recon_img_path)

                        if enable_wandb:
                            wandb.log({'Latents/wm{}'.format(j): wandb.Image(latent_img)}, step=n)
                            wandb.log({f'Reconstructions/m{j}': wandb.Image(reconstructions[j].mean)}, step=n)

                    latent_img = latents[:, -latent_dim_z:].detach().cpu()
                    latent_img = torch.transpose(latent_img, 0, 1).reshape(-1, *latent_img.shape[2:]).unsqueeze(1)
                    latent_img = make_grid(latent_img, nrow=gen_batch_size)

                    latent_img_path = os.path.join(runPath, f"batch_{n}_latents_z.png")
                    tvu.save_image(latent_img, latent_img_path, normalize=True)

                    if enable_wandb:
                        wandb.log({'Latents/z': wandb.Image(latent_img)}, step=n)

                    if (n + 1) * conf.training.batch_size >= n_samples:
                        break


        def test_fid(epoch, n_samples=32, use_diffusion=True):
            gen_batch_size = 8
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
                fid_score = fid(tempdirs[mod], dataset_name=f"nyu_{mod}", dataset_res=299, dataset_split="test")
                if enable_wandb:
                    wandb.log({f"FID_{mod}": fid_score}, step=epoch)

                fid_scores.append(fid_score)

                print(f'====>             FID {mod}: {fid_score:.2f}')

            return fid_scores


        with Timer('MMVAEplus') as t:
            visualise_generated_latents(3)

    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


@hydra.main(version_base=None, config_path="configs/training/", config_name="nyu_depth_stable_diffusion_train.yaml")
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

