# Train MMVAEplus on CUB Image-Captions dataset
import os

import pandas as pd
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
from torchvision.transforms import v2, InterpolationMode
from torchvision.utils import make_grid
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.utils as tvu

from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
from ldm.util import instantiate_from_config, Timer
from stablediffusion.objectives.mmvae_objectives import m_dreg
from ldm.data import get_dataset

from utils import save_and_log_latents, save_and_log_img

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
            run = wandb.init(
                # Set the project where this run will be logged
                project=conf.experiment,
                # Track hyperparameters and run metadata
                config=OmegaConf.to_container(conf),
                # Run name
                name=f"{runId}_{str(rank)}",
                group=f"DDP_{runId}",
                notes=exp_notes,
            )
        else:
            run = None

        modalities = conf.data.modalities
        kwargs = {'num_workers': 4, 'pin_memory': True} if not data_parallel else {"num_workers": 0, "pin_memory": True}
        tx = {"colour": v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)]),
              "depth": v2.Compose([v2.Resize(conf.data.input_size), v2.CenterCrop(conf.data.input_size)]),
              "semseg": v2.Compose([v2.Resize(conf.data.input_size, interpolation=InterpolationMode.NEAREST_EXACT),
                                    v2.CenterCrop(conf.data.input_size)])}

        test_dataset = get_dataset(conf.data.dataset_name,
                                   root_data_dir=conf.data.datadir,
                                   transform=tx,
                                   split="val",
                                   modality=modalities)

        display_data_fs = [getattr(test_dataset, f"{mode}_display") for mode in modalities]

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank,
                                          shuffle=False) if data_parallel else None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.training.batch_size, shuffle=False,
                                                  sampler=test_sampler, **kwargs)

        def unpack_data_nyu(data, device="cuda"):
            return [d.to(device) for d in data]

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

        def visualise_latents(n_samples, avg_z=True):
            vaes = model.module.module.vaes if data_parallel else model.module.vaes

            latent_dim_w = model.module.module.w_channels if data_parallel else model.module.w_channels
            latent_dim_z = model.module.module.params.z_channels if data_parallel else model.module.z_channels

            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = unpack_data_nyu(dataT, device=device)

                    for m, name in enumerate(modalities):
                        save_and_log_img(data[m].detach().cpu(), f'Input/m{m}', runPath, display_data_fs[m], i, run)

                    latents = model.module("encode", data, avg_z=avg_z) if data_parallel else model("encode", data,
                                                                                                    avg_z=avg_z)

                    if avg_z:
                        reconstructions = model.module("decode", latents) if data_parallel else model("decode", latents)

                        for j in range(NUM_VAES):
                            latent_img = latents[:, j:j + latent_dim_w]

                            save_and_log_latents(latent_img.detach().cpu(), f'Latents/wm{j}', runPath, i, run)
                            save_and_log_img(reconstructions[j], f'Reconstructions/m{j}',
                                             runPath, display_data_fs[j], i, run)

                        latent_img = latents[:, -latent_dim_z:]
                        save_and_log_latents(latent_img, 'Latents/z', runPath, i, run)
                    else:
                        for j in range(NUM_VAES):  # Choice of z
                            latent_w_img = latents[j]
                            latent_z_img = latents[j + NUM_VAES]

                            save_and_log_latents(latent_w_img, f'Latents/wm{j}', runPath, i, run)
                            save_and_log_latents(latent_z_img, f'Latents/zm{j}', runPath, i, run)

                            # Reconstructions with known w
                            ws = latents[:NUM_VAES]
                            u = torch.cat([*ws, latent_z_img], dim=1)
                            reconstructions = model.module("decode", u) if data_parallel else model("decode", u)

                            for k in range(NUM_VAES):
                                save_and_log_img(reconstructions[k], f'Reconstructions/z{j}/m{k}',
                                                 runPath, display_data_fs[k], i, run)

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
                                                 runPath, display_data_fs[k], i, run)

                    if (i + 1) * conf.training.batch_size >= n_samples:
                        break

        def check_z_closeness(n_samples=10):
            with torch.no_grad():

                z_mses = []
                z_mags = [[] for _ in range(NUM_VAES)]
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), position=rank if data_parallel else 0):
                    data = unpack_data_nyu(dataT, device=device)

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

        def check_learned_prior_params():
            vaes = model.module.module.vaes if data_parallel else model.module.vaes
            for m, vae in enumerate(vaes):
                pw_params = vae.pw_params_aux

                mean_path = os.path.join(runPath, f"vae{m}_pw_mean_aux.png")
                var_path = os.path.join(runPath, f"vae{m}_pw_var_aux.png")

                mean = pw_params[0].detach().cpu().transpose(0, 1)
                var = pw_params[1].detach().cpu().transpose(0, 1)

                print(f"VAE {m} pz average variance {var.mean()}, min: {var.min()}, max: {var.max()}")

                tvu.save_image(mean, mean_path, normalize=True)
                tvu.save_image(var, var_path, normalize=True)
                if enable_wandb:
                    wandb.log({f'pw_params/vae{m}/mean': wandb.Image(mean)}, step=0)
                    wandb.log({f'pw_params/vae{m}/var': wandb.Image(var)}, step=0)

            pw_params_global = model.module.pw_params

            mean_path = os.path.join(runPath, f"global_pw_mean_aux.png")
            var_path = os.path.join(runPath, f"global_pw_var_aux.png")

            mean = pw_params_global[0].detach().cpu().transpose(0, 1)
            var = pw_params_global[1].detach().cpu().transpose(0, 1)

            print(f"global pw average variance {var.mean()}, min: {var.min()}, max: {var.max()}")

            tvu.save_image(mean, mean_path)
            tvu.save_image(var, var_path)
            if enable_wandb:
                wandb.log({f'pw_params/global/mean': wandb.Image(mean)}, step=0)
                wandb.log({f'pw_params/global/var': wandb.Image(var)}, step=0)

            pz_params_global = model.module.pz_params

            mean_path = os.path.join(runPath, f"global_pz_mean_aux.png")
            var_path = os.path.join(runPath, f"global_pz_var_aux.png")

            mean = pz_params_global[0].detach().cpu().transpose(0, 1)
            var = pz_params_global[1].detach().cpu().transpose(0, 1)

            print(f"global pz average variance {var.mean()}, min: {var.min()}, max: {var.max()}")

            tvu.save_image(mean, mean_path)
            tvu.save_image(var, var_path)
            if enable_wandb:
                wandb.log({f'pz_params/global/mean': wandb.Image(mean)}, step=0)
                wandb.log({f'pz_params/global/var': wandb.Image(var)}, step=0)


        def decode_encode_latent():
            # Note this is a simple test designed for unimodal data (ie StableIndigo work)
            # - I should make a different file but im a little lazy
            vaes = model.module.module.vaes if data_parallel else model.module.vaes


            with torch.no_grad():
                for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                                          position=rank if data_parallel else 0):
                    data = unpack_data_nyu(dataT, device=device)

                    latents = model.module("encode", data, avg_z=avg_z) if data_parallel else model("encode", data,
                                                                                                    avg_z=avg_z)

                    if avg_z:
                        reconstructions = model.module("decode", latents) if data_parallel else model("decode", latents)



        with Timer('MMVAEplus') as t:
            # check_z_closeness(16)
            visualise_latents(64, avg_z=False)
            # check_learned_prior_params()


    finally:
        if data_parallel:
            print("Performing cleanup")
            cleanup()


exp_notes = "Show the auxilliary (learned for each VAE) and global priors on w and z"

"""
("Test the cross generation capabilities particularly using the individual z encodings. "
         "Test self and cross generation with either z and using sampled (random w reconstrunction) or encoded w values (reconstruction)."
         "For encoded values the z encoding used is the batch size modulo 2. For the random w generation we use the global pw parameters not"
         " the local ones learned by the VAE, though I dont understand why there are two versions")
         """

@hydra.main(version_base=None, config_path="configs/testing/", config_name="celeba_semseg_autoencoder_test.yaml")
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

