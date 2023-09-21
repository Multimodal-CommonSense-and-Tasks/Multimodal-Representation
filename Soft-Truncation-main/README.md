## Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation
**[Dongjun Kim](https://sites.google.com/view/dongjun-kim) \*,[Seungjae Shin](https://sites.google.com/view/seungjae-shin) \*, Kyungwoo Song, Wanmo Kang, Il-Chul Moon**

| [paper](http://arxiv.org/abs/2106.05527) | <br>

This repo contains an official PyTorch implementation for the paper [Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation](http://arxiv.org/abs/2106.05527).

## Overview

We propose a genearlly applicable training method for a general weighted diffusion loss.

![schematic](figure/sample_figures_256.jpg)

Abstract: Recent advances in diffusion models bring stateof-the-art performance on image generation tasks. However, empirical results from previous research in diffusion models imply an inverse correlation between density estimation and sample generation performances. This paper investigates with sufficient empirical evidence that such inverse correlation happens because density estimation is significantly contributed by small diffusion time, whereas sample generation mainly depends on large diffusion time. However, training a score network well across the entire diffusion time is demanding because the loss scale is significantly imbalanced at each diffusion time. For successful training, therefore, we introduce Soft Truncation, a universally applicable training technique for diffusion models, that softens the fixed and static truncation hyperparameter into a random variable. In experiments, Soft Truncation achieves stateof-the-art performance on CIFAR-10, CelebA, CelebA-HQ 256 × 256, and STL-10 datasets.

Idea and Contribution

Code & experiment

## Running Commands

**CIFAR-10**

- DDPM++ (VP, NLL) + ST

```shell script
python main.py --config configs/vp/CIFAR10/ddpmpp_nll_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

- DDPM++ (VP, FID) + ST

```shell script
python main.py --config configs/vp/CIFAR10/ddpmpp_fid_st_deepest.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

- UNCSN++ (RVE) + ST

```shell script
python main.py --config configs/ve/CIFAR10/uncsnpp_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

**CelebA**

- DDPM++ (VP, NLL) + ST

```shell script
python main.py --config configs/vp/CELEBA/ddpmpp_nll_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

- DDPM++ (VP, FID) + ST

```shell script
python main.py --config configs/vp/CELEBA/ddpmpp_fid_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

- UNCSN++ (RVE) + ST

```shell script
python main.py --config configs/ve/CELEBA/uncsnpp_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

**ImageNet32**

- DDPM++ (VP, NLL) + ST

```shell script
python main.py --config configs/vp/IMAGENET32/ddpmpp_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

**CelebA-HQ**

- UNCSN++ (RVE) + ST

```shell script
python main.py --config configs/ve/celebahq/uncsnpp_st.py --workdir YOUR_SAVING_DIRECTORY --mode train
```

## Pretrained checkpoints
We release our checkpoints [here](https://drive.google.com/drive/folders/1Wyk0ucFW-QDS_g1EcPm361LWWgWqJ6L_).

## Experimental Results


## References

If you find the code useful for your research, please consider citing
```bib
@article{kim2021soft,
  title={Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation},
  author={Kim, Dongjun and Shin, Seungjae and Song, Kyungwoo and Kang, Wanmo and Moon, Il-Chul},
  journal={arXiv e-prints},
  pages={arXiv--2106},
  year={2021}
}
```
This work is heavily built upon the code from
* Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." *International Conference on Learning Representations (ICLR)*. 2021.
* Song, Yang, et al. "Maximum likelihood training of score-based diffusion models." *Advances in Neural Information Processing Systems (NeurIPS)*. 2021.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems (NeurIPS)*. 2020.
