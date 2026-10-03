# Multi-Modal Latent Diffusion using Information Disentangling Autoencoders

This repository contains code to accompany the paper [*Multi-Modal Latent Diffusion using Information Disentangling Autoencoders*](https://arxiv.org/abs/2211.13227), and includes training and inference scripts for both autoencoders and latent diffusion models.
The code is based on [Stable Diffusion](https://github.com/compvis/stable-diffusion) and [MMVAE]() repositories.
  
## Requirements

To install the required dependencies, we recommend creating a new conda environment and running the following commands from the root of the repository.

```
conda env create -f environment.yaml
conda activate diffusion_multimodal
``` 

## Datasets

The datasets used in the paper can be downloaded from the following links:

 - [CelebAMask-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html)
 - [LSUN Bedroom](https://github.com/fyu/lsun/tree/master)
 - [NYUv2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/)

We recommend using the (webdataset)[https://github.com/webdataset/webdataset] format for large scale datasets and provide a dataset class in `ldm/data/dataset_webformat.py` for this purpose. 
The NYUv2 dataset is already provided in this format. We provide a script `scripts\construct_lsun_web_dataset.sh` to convert LSUN to webdataset format. 

As CelebAMask-HQ is a smaller dataset we use it directly with the dataset class from `ldm/data/dataset_CelebAMask.py`.

## Training

### Autoencoder

The script `train_mm_autoencoder.py` handles both training and testing of the multimodal autoencoder. Training follows two stages: first training encoders and decoders, then finetuning decoders with the additional shared latent combiner network.
This can be achieved with the following example commands:

##### Stage 1: Train encoders and decoders
```
# celeba RGB + semantic segmentation
python train_mm_autoencoder.py -cn celeba_semseg_autoencoder_train

# LSUN Bedroom RGB + Depth
python train_mm_autoencoder.py -cn nyu_depth_autoencoder_train
```


##### Stage 2: Finetune decoders with the shared latent combiner
```
# celeba RGB + semantic segmentation
python train_mm_autoencoder.py -cn celeba_semseg_autoencoder_train_z_combiner

# LSUN Bedroom RGB + Depth
python train_mm_autoencoder.py -cn nyu_depth_autoencoder_train_z_combiner
```

### Diffusion

The scripts `train_mm_diffusion.py` handles latent diffusion training and testing, given the trained autoencoder.

```
# celeba RGB + semantic segmentation
python train_mm_diffusion.py -cn celeba_semseg_diffusion_train

# LSUN Bedroom RGB + Depth
python train_mm_diffusion.py -cn nyu_depth_lsun_stable_diffusion_train
```

 We adapt the adjustable EMA developed by NVIDIA, therefore after training different levels of EMA can be averaged using the following command

```
# Train latent diffusion model
python train_mm_diffusion.py -cn celeba_semseg_diffusion_train_optimise_ema
```

### Evaluation

To compute metrics from our paper the relevant FID, KID and CMMD reference statistics must be computed first. This can be done using the following commands:

```
python -m scripts.make_celeba_metric_reference
python -m scripts.make_nyu_lsun_metric_reference
```

Metrics can then be computed with the pyiqa library.

## Cite

If you find our work useful please consider citing our paper:

```

```


