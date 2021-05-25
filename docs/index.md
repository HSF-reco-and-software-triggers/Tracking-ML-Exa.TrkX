# Machine Learning for Particle Track Reconstruction

Welcome to repository and documentation for ML pipelines and techniques by the ExatrkX Collaboration. Here we present a set of templates, best practices and results gathered from significant trial and error, to speed up the development of others in the domain of machine learning for high energy physics. We focus on applications specific to detector physics, but many tools can be applied to other areas, and these are collected in an application-agnostic way in the [Tools](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/tools/overview/) section.

## Intro

To start as quickly as possible, clone the repository, [Install](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart) and follow the steps in [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart). This will get you generating toy tracking data and running inference immediately. Many of the choices of structure will be made clear there. If you already have a particle physics problem in mind, you can apply the [Template](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/choosingguide.md) that is most suitable to your use case.

Once up and running, you may want to consider more complex ML [Models](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/models/overview/). Many of these are built on other libraries (for example [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)).

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/application_diagram_1.png" width="600"/>
</figure>
</div>

## Install

It's recommended to start a conda environment before installation:

```
conda create --name exatrkx-tracking python=3.8
conda activate exatrkx-tracking
pip install pip --upgrade
```

If you have a CUDA GPU available, load the toolkit or [install it](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) now.  

```
python install.py
```

will **attempt** to negotiate a path through the packages required, using `nvcc --version` to automatically find the correct wheels. **Warning**: If you are installing with cpu, this may take up to 15 minutes due to an unfortunately slow installation of Pytorch3D from source.

You should be ready for the [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart)!

If this doesn't work, you can step through the process manually:

<table style="border: 1px solid gray; border-collapse: collapse">
<tr style="border-bottom: 1px solid gray">
<th style="border-bottom: 1px solid gray"> CPU </th>
<th style="border-left: 1px solid gray"> GPU </th>
</tr>
<tr>
<td style="border-bottom: 1px solid gray">

1. Run 
`export CUDA=cpu`
    
</td>
<td style="border-left: 1px solid gray">

1a. Find the GPU version cuda XX.X with `nvcc --version`
    
1b. Run `export CUDA=cuXXX`, with `XXX = 92, 101, 102, 110`

</td>
</tr>
<tr style="border-bottom: 1px solid gray">
<td colspan="2">

2. Install Pytorch and dependencies 

```pip install --user -r requirements.txt```

</td>
</tr>
<tr style="border-bottom: 1px solid gray">
<td colspan="2">

3. Install local packages

```pip install -e .```
    
</td>
</tr>
<tr>
<td style="border-bottom: 1px solid gray">

4. Install CPU-optimized packages

```pip install faiss-cpu
 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"``` 
    
</td>
<td style="border-left: 1px solid gray">

4. Install GPU-optimized packages

```pip install faiss-gpu cupy-cudaXXX```, with `XXX`
    
```pip install pytorch3d 
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3{Y}_cu{XXX}_pyt{ZZZ}/download.html```
    
where `{Y}` is the minor version of Python 3.{Y}, `{XXX}` is as above, and `{ZZZ}` is the version of Pytorch {Z.ZZ}.

    e.g. `py36_cu101_pyt170` is Python 3.6, Cuda 10.1, Pytorch 1.70.
    
</td>
</tr>
</table>