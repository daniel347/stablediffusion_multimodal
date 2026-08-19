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
 - [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

We also provide a dataset class in `ldm/data/dataset_webformat.py` for loading datasets in the (webdataset)[https://github.com/webdataset/webdataset]
 which we recommend for training on large datasets.

## Training

### Autoencoder

The script `train_mm_autoencoder.py` handles both training and testing of the multimodal autoencoder. Training follows two stages: first training encoders and decoders, then finetuning decoders with the additional shared latent combiner network.
This can be achieved with the following example commands:

```
# Stage 1: Train encoders and decoders
python train_mm_autoencoder.py -cn celeba_semseg_autoencoder_train

# Stage 2: Finetune decoders with the shared latent combiner
python train_mm_autoencoder.py -cn celeba_semseg_autoencoder_train_z_combiner
```



## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


