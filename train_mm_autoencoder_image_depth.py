# Train MMVAEplus on CUB Image-Captions dataset
import os
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import optim
from torchvision.transforms import v2
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config, dict_to_namespace, namespace_to_dict, Timer, save_model_light
import hydra
from omegaconf import OmegaConf, DictConfig
from mmvae_objectives import m_dreg
from functools import partial
from torch.profiler import ProfilerActivity, profile, record_function
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler, ExponentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from ldm.models.mmvae_data_parallel_wrapper import MMVAEDataParallelWrapper
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import time


from collections.abc import Iterable
import copy
from datetime import datetime

import psutil

from ldm.data.dataset_NYU_Depth import NYUDepth

import torchvision.utils as tvu

import wandb


def expand_sub_confs(conf):
    # Expand sub-configs
    for key, val in conf.items():
        if key[-4:] == "conf":
            subconf = OmegaConf.load(val)
            conf[key] = expand_sub_confs(subconf)
    return conf


@hydra.main(version_base=None, config_path="configs/training/", config_name="nyu_depth_autoencoder_train.yaml")
def main(conf: DictConfig) -> None:
    torch.cuda.memory._record_memory_history()
    conf = expand_sub_confs(conf)

    # Random seed
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    # CUDA stuff
    data_parallel = False
    if isinstance(conf.device, str):
        device = torch.device(conf.device)
    elif isinstance(conf.device, Iterable):
        device = torch.device(conf.device[0])
        data_parallel = True

    # Initialise with SD model?
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


    if data_parallel:
        print(f"Initialising data-based parallelism with {len(conf.device)} GPUs")
        model = DDP(model, device_ids=conf.device)

    model = model.to(device)

    # Set experiment name if not set
    if not conf.experiment:
        conf.experiment = model.modelName

    # Set up run path
    runId = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    experiment_dir = Path(os.path.join(conf.data.outputdir, conf.experiment, "checkpoints"))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = os.path.join(str(experiment_dir), runId)
    if os.path.exists(runPath):
        shutil.rmtree(runPath)
    os.makedirs(runPath)
    print('Expt:', runPath)
    print('RunID:', runId)

    NUM_VAES = len(model.module.module.vaes) if data_parallel else len(model.module.vaes)

    # save args to run
    with open('{}/args.json'.format(runPath), 'w') as fp:
        OmegaConf.save(config=conf, f=fp)

    # WandB
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=conf.experiment,
        # Track hyperparameters and run metadata
        config=OmegaConf.to_container(conf),
        # Run name
        name=runId
    )

    # preparation for training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=conf.training.lr, amsgrad=True)

    scheduler = ChainedScheduler([LinearLR(optimizer, 0.1, 1, total_iters=10),
                                  ExponentialLR(optimizer, 0.99)])

    # Load NYU dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if device == "cuda" else {}
    tx = v2.Compose([v2.Resize([256, 256])])

    train_dataset = NYUDepth(conf.data.datadir, transform=tx, train=True, modality=['colour', 'depth'])
    test_dataset = NYUDepth(conf.data.datadir, transform=tx, train=False, modality=['colour', 'depth'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.training.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.training.batch_size, shuffle=False, **kwargs)

    # Set the learning rate forward to continue training after pause
    if conf.training.start_epoch > 0:
        for _ in range(len(train_loader)//100):
            scheduler.step()

    def unpack_data_nyu(data, device="cuda"):
        return [data[0].to(device), data[1].to(device)]

    objective = partial(m_dreg, beta=conf.training.beta)
    t_objective = objective


    def train(epoch):
        model.train()
        b_loss = 0

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for i, dataT in pbar:
            data = unpack_data_nyu(dataT, device=device)
            optimizer.zero_grad()
            loss = -objective(model, data, K=conf.training.K)
            loss.backward()
            optimizer.step()
            b_loss += loss.detach().item()
            if conf.print_freq > 0 and i % conf.print_freq == 0:
                pbar.set_description("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / conf.training.batch_size))
                """
                t_now = time.time()
                t_diff = t_now - last_t_elapsed
                last_t_elapsed = t_now
                print(f"last {conf.print_freq} iterations took {t_diff//3600}:{t_diff//60 - (t_diff//3600)*60}:{t_diff - (t_diff//60)*60:.2f}")
                """

            """
            if i == 10:
                torch.memory._dump_snapshot("mmvae_cpu_train_checkpoint_memory.pk")
                raise Exception("Memory dump")
            """

            if i % 100 == 0:
                scheduler.step()

            """
            cpu_mem_avail.append(psutil.virtual_memory().available)

        plt.plot(cpu_mem_avail)
        plt.ylabel("memory available")
        plt.savefig("cpu_mem_train.png")
        plt.clf()

        raise Exception("Memory dump")
        """

        # Epoch loss
        epoch_loss = b_loss / len(train_loader.dataset)
        wandb.log({"Loss/train": epoch_loss}, step=epoch)
        print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, epoch_loss))


    def test(epoch):
        b_loss = 0
        with torch.no_grad():
            for i, dataT in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                """
                if i == 3:
                    torch.cuda.memory._dump_snapshot("mmvae_test_memory.pk")
                    raise Exception("Memory dump")
                """

                data = unpack_data_nyu(dataT, device=device)
                loss = -t_objective(model, data, K=conf.training.K, test=True)
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
                            wandb.log({'Cross_Generation/m{}/m{}'.format(i, j): wandb.Image(cg_imgs[i][j])}, step=epoch)
                    del cg_imgs

                """
                t_now = time.time()
                t_diff = t_now - last_t_elapsed
                last_t_elapsed = t_now
                print(f"last {conf.print_freq} iterations took {t_diff//3600}:{t_diff//60 - (t_diff//3600)*60}:{t_diff - (t_diff//60)*60:.2f}")
                """


        # Epoch test loss
        epoch_loss = b_loss / len(test_loader.dataset)
        wandb.log({"Loss/test": epoch_loss}, step=epoch)
        print('====>             Test loss: {:.4f}'.format(epoch_loss))


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

    with Timer('MMVAEplus') as t:
        test(0)
        gen_samples = model("generate_unconditional", N=4)
        for j in range(NUM_VAES):
            wandb.log({'Generations/m{}'.format(j): wandb.Image(gen_samples[j])}, step=0)
        del gen_samples

        for epoch in range(conf.training.start_epoch + 1, conf.training.num_epochs + 1 + conf.training.start_epoch):
            train(epoch)
            if epoch % 1 == 0:
                test(epoch)
                gen_samples = model("generate_unconditional", N=4)
                for j in range(NUM_VAES):
                    wandb.log({'Generations/m{}'.format(j): wandb.Image(gen_samples[j])}, step=epoch)
                del gen_samples
            if epoch % 1 == 0:
                save_model_light(model, runPath + '/model_'+str(epoch)+'.rar')


if __name__ == '__main__':
    main()

