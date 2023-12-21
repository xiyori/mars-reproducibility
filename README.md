Forked from https://github.com/MaxBourdon/mars/tree/main

# MARS: Masked Automatic Ranks Selection in Tensor Decompositions
This repository contains code for our paper [MARS: Masked Automatic Ranks Selection in Tensor Decompositions](https://arxiv.org/abs/2006.10859).


The main files are:
* mars.py &mdash; the main module, containing realizations of the MARS wrapper over a tensorized model, the MARS loss and auxiliary functions;
* tensorized_models.py &mdash; module, containing realizations of several implemented tensorized models, the base class and auxiliary functions.

The notebooks are:
* MNIST-2FC-soft.ipynb &mdash; Jupyter Notebook, replicating the MNIST 2FC-Net experiment using soft compression mode;
* MNIST-2FC-hard.ipynb &mdash; Jupyter Notebook, replicating the MNIST 2FC-Net experiment using hard compression mode.
* VAE-AE-Baseline.ipynb &mdash; autoencoder and variational autoencoder template of baseline for further experiments
  
* MNIST-AE.ipynb &mdash; Jupyter Notebook, Factorized autoencoder
* MNIST-VAE.ipynb &mdash; Jupyter Notebook, Factorized variational autoencoder
* MNIST-VAE-TT.ipynb &mdash; Jupyter Notebook, Successful application of tensor train to variational autoencoder
  
* CIFAR10-ResNet-naive.ipynb &mdash; Jupyter Notebook, ResNet-110 on CIFAR10
* CIFAR10-ResNet-base.ipynb &mdash; Jupyter Notebook, ResNet-110 on CIFAR10
* CIFAR10-ResNet-proper.ipynb &mdash; Jupyter Notebook, ResNet-110 on CIFAR10

* MNIST-LeNet-base.ipynb &mdash; Jupyter Notebook, LeNet-5 on MNIST
* MNIST-LeNet-compress.ipynb &mdash; Jupyter Notebook, LeNet-5 on MNIST
  

To run the notebooks, first, install the tt-pytorch library from https://github.com/KhrulkovV/tt-pytorch  
System requirements and dependencies are described in https://github.com/KhrulkovV/tt-pytorch/blob/master/README.md  
After installing all the dependencies, run the following command to install tt-pytorch from Git via pip: `pip install git+https://github.com/KhrulkovV/tt-pytorch.git`

Our team:

@sspetya - Petr Sychev

@gurkwe - Petr Kushnir

@xiyori - Foma Shipilov

@MarioAuditore - Elfat Sabitov

@skushneryuk - Sergey Kushneryuk
