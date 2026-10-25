# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Perform post-hoc EMA reconstruction."""

import os
import copy
import warnings
import torch
import ldm.modules.ema as ema
import glob
import tqdm
import logging
import pandas as pd
from pathlib import Path
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Perform post-hoc EMA reconstruction.
# Returns an iterable that yields dnnlib.EasyDict(out, step_idx, num_steps),
# where 'out' is a list of dnnlib.EasyDict(net, nimg, std, pkl_data, pkl_path)

def reconstruct_phema(
    in_folder,                    # List of input pickles, expressed as dnnlib.EasyDict(path, nimg, std).
    out_std,                    # List of relative standard deviations to reconstruct.
    out_nimg=None,     # Training time of the snapshot to reconstruct. None = highest input time.
    out_dir=None,     # Where to save the reconstructed network pickles. None = do not save.
    epoch_size=24000,   # number of images per epoch
):
    if (Path(in_folder) / "saves.csv").exists():
        save_df = pd.read_csv(os.path.join(in_folder, "saves.csv"))
        save_df = save_df[save_df.iteration > 0]  # Ignore model saves with 0
        saves_iteration = save_df["iteration"].tolist()
        saves_iteration = [s * epoch_size for s in saves_iteration]
        in_std = save_df["std"].tolist()
        save_path = save_df["path"].tolist()
        ema_ind = save_df["ema_ind"].tolist()
    else:
        saves = glob.glob(f"{in_folder}/model_*.pt")

        saves_iteration = []
        in_std = []
        save_path = []
        ema_ind = []

        logger.info("Extracting information from the saves...")
        for save in tqdm.tqdm(saves):
            # expand the saves to extract std/iteration information for each ema
            save_dict = torch.load(save, map_location='cpu')
            print(save_dict.keys())
            if save_dict["iteration"] == 0:
                # I don't know how we even get model_0 saved
                continue
            for i, std in enumerate(save_dict["ema"]["stds"]):
                in_std.append(std)
                saves_iteration.append(save_dict["iteration"])
                ema_ind.append(i)
                save_path.append(save)

            del save_dict

        save_df = pd.DataFrame({"std": in_std, "iteration": saves_iteration, "path": save_path, "ema_ind": ema_ind})
        save_df.to_csv(Path(in_folder) / "saves.csv")

        saves_iteration = [s * epoch_size for s in saves_iteration]

    if out_nimg is None:
        out_nimg = [max(saves_iteration)]

    for oni in out_nimg:
        if oni not in saves_iteration:
            raise ValueError(f"Reconstruction time {oni} not found in input saves.")

    out_std_batch = out_std

    # Return an iterable over the reconstruction steps.
    class ReconstructionIterable:
        def __len__(self):
            return len(saves_iteration)

        def __iter__(self):
            # Loop over batches.
            coefs = ema.solve_posthoc_coefficients(saves_iteration, in_std, out_nimg, out_std_batch)
            logger.debug(f"Obtained coefficients: {coefs}")
            out = [None for _ in out_std_batch]  # {"model": None, "iteration": None, "ema": None}

            # Loop over input pickles.
            for i in range(len(save_path)):
                sd = torch.load(save_path[i], map_location='cpu')
                in_net = sd['ema']["emas"][ema_ind[i]]

                # Accumulate weights for each output pickle.
                for j in range(len(out)):
                    if out[j] is None:
                        out[j] = {"model": copy.deepcopy(sd["model"]), "iteration": out_nimg[j]}
                        for _, pj in out[j]["model"].items():
                            pj.zero_()
                    for pi, pj in zip(in_net.values(), out[j]["model"].values()):
                        if torch.any(torch.isnan(pi)):
                            print(f"found NaN parameter in input model {j} at iteration {i}")
                        if isinstance(pi, torch.Tensor):
                            # logger.debug(f"Copying parameter of shape {pi.shape} to {pj.shape}")
                            pj += pi.to(torch.float32) * coefs[i, j]
                        else:
                            logger.debug(f"found non parameter of type {type(pi)}, copying without weighting")
                            pj.copy_(pi.to(torch.float32))
                        if torch.any(torch.isnan(pj)):
                            print(f"found NaN parameter in output model {j} at iteration {i}")

                # Finalise outputs.
                if i == len(save_path) - 1:
                    for j in range(len(out)):
                        """
                        for _, pj in out[j]["model"].items():
                            pj.to(torch.float16)
                        """
                        out_path = os.path.join(out_dir, f"model_it{out_nimg[j]//epoch_size}_ema_{out_std_batch[j]:.3f}.pt")
                        torch.save(out[j], out_path)

                # Yield results.
                del sd, in_net # conserve memory
                yield

    return ReconstructionIterable()



if __name__ == "__main__":
    saves_path = "path/to/saves"
    it = reconstruct_phema(saves_path, out_std=[0.2], out_nepoch=[10], out_dir=saves_path, epoch_size=24000)
    for _ in it:
        pass
