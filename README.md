<div align="center">

# Tracking with ML

<figure>
    <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/final_wide.png" width="250"/>
</figure>
    
### Exa.TrkX Collaboration


[Documentation](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/)

[![make_docs](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/actions/workflows/make_docs.yml/badge.svg)](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/actions/workflows/make_docs.yml) [![ci](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/actions/workflows/ci.yml/badge.svg)](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/actions/workflows/ci.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Lint](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/actions/workflows/black.yml/badge.svg)](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/actions/workflows/black.yml)


</div>

Welcome to repository and documentation for ML pipelines and techniques by the ExatrkX Collaboration. 

## Objectives

1. To present the ExaTrkX Project pipeline for track reconstruction of TrackML and ITk data in a clear and simple way
2. To present a set of templates, best practices and results gathered from significant trial and error, to speed up the development of others in the domain of machine learning for high energy physics. We focus on applications specific to detector physics, but many tools can be applied to other areas, and these are collected in an application-agnostic way in the [Tools](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/tools/overview/) section.

## Intro

To start as quickly as possible, clone the repository, [Install](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/#install) and follow the steps in [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart). This will get you generating toy tracking data and running inference immediately. Many of the choices of structure will be made clear there. If you already have a particle physics problem in mind, you can apply the [Template](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/choosingguide.md) that is most suitable to your use case.

Once up and running, you may want to consider more complex ML [Models](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/models/overview/). Many of these are built on other libraries (for example [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)).

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/application_diagram_1.png" width="600"/>
</figure>
</div>

## Install

Thank god that almost every package used in this library is available on conda and has known compatibility for both CPU and GPU (with CUDA v11.3). Therefore installation should be as simple as:

**CPU-only installation**
```
conda env create -f cpu_environment.yml python=3.9
conda activate exatrkx-cpu
pip install -e .
```

Or, after ensuring your GPU drivers are [updated to run CUDA v11.3](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html):

**GPU installation**
```
conda env create -f gpu_environment.yml python=3.9
conda activate exatrkx-gpu
pip install -e .
```


You should be ready for the [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart)!

### Vintage Errors

A very possible error will be
```
OSError: libcudart.so.XX.X: cannot open shared object file: No such file or directory
```
This indicates a mismatch between CUDA versions. Identify the library that called the error, and ensure there are no versions of this library installed in parallel, e.g. from a previous `pip --user` install.