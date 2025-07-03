import torch
import cmmd_pytorch as cmmd
import glob
import numpy as np
from pathlib import Path
import shutil
from ldm.data.dataset_CelebAMask import MMCelebA
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modalities = ["colour"]

generation_path = "/rds/general/user/ds3119/ephemeral/outputs/MM_CelebA_Stable_Diffusion/checkpoints/2025_07_02__16_51_12_sample_latents_diff_ema_opt_metrics/temp_fid"
embedding_dir = "../../python_modules/cmmd_pytorch/cache"

dataset = MMCelebA(root_data_dir="../../datasets/CelebAMask-HQ",
                   split='train', transform=None,
                   modality=["colour", "semseg"])

data_path_dir = {mode: [os.path.join(dataset.img_dir, f"{index}.jpg") for index in dataset.index_list] for mode in modalities}

out_dir = Path("memorisation_check")
out_dir.mkdir(parents=True, exist_ok=True)

for mode in modalities:
    dataset_clip_embeddings = f"{embedding_dir}/nyu_depth_lsun_{mode}_val_299.npy"
    data_paths = data_path_dir[mode]

    mode_gen_path = f"{generation_path}_{mode}"
    mode_gen_files = list(glob.glob(mode_gen_path + "/*"))

    cmmd.compute_and_cache_embeddings(dir=None,
                                      files=mode_gen_files,
                                      dataset_name="nyu_depth_lsun_colour_gen",
                                      dataset_res=299,
                                      dataset_split="val",
                                      device=device)

    gen_clip_embeddings = f"{embedding_dir}/nyu_depth_lsun_{mode}_gen_val_299.npy"

    gen_emb = np.load(gen_clip_embeddings)
    data_emb = np.load(dataset_clip_embeddings)

    n_sim = 3
    sim_thresh = 0.975

    for i, ge in enumerate(gen_emb):
        data_cos_sim = data_emb @ ge
        max_n_inds = np.argpartition(data_cos_sim, -n_sim)[-n_sim:]
        print(f"top {n_sim} similarities: {data_cos_sim[max_n_inds]}")
        if np.any(data_cos_sim[max_n_inds] > sim_thresh):
            print("Generation is similar to data!")
            data_paths = data_paths[max_n_inds]
            dst = out_dir / f"{mode}_{i}"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(mode_gen_files[i], dst / f"{i}_gen.png")
            for j, p in enumerate(data_paths):
                dst_path = dst / f"{j}_sim_{data_cos_sim[max_n_inds[j]]:.2f}.png"
                shutil.copy(p, dst)
