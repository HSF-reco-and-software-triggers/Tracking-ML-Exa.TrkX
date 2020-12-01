# Machine Learning for Particle Track Reconstruction

Welcome to repository and documentation for ML pipelines and techniques by the ExatrkX Collaboration. Here we present a set of templates, best practices and results gathered from significant trial and error, to speed up the development of others in the domain of machine learning for high energy physics. We focus on applications specific to detector physics, but many tools can be applied to other areas, and these are collected in an application-agnostic way in the [Tools](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/tools/overview/) section.

## Intro

To start as quickly as possible, clone the repository and follow the steps in [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart.md). This will get you generating toy tracking data and running inference immediately. Many of the choices of structure will be made clear there. If you already have a particle physics problem in mind, you can apply the [Template](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/choosingguide.md) that is most suitable to your use case.

Once up and running, you may want to consider more complex ML [Models](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/models/overview/). Many of these are built on other libraries (for example [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)).

<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/application_diagram_1.png"/>
  <figcaption>The repository through an application lens, using Pipelines for specific physics goals</figcaption>
</figure>

## Install

The repository can be installed and run with GPU or CPU. The installation depends on this compatibility:

<table style="border: 1px solid gray">
<tr>
<th> CPU </th>
<th> GPU </th>
</tr>
<tr>
<td>

1. Run 
`export CUDA=cpu`
    
</td>
<td>

1a. Find the GPU version cuda XX.X with `nvcc --version`
    
1b. Run `export CUDA=cuXXX`, with `XXX = 92, 101, 102, 110`

</td>
</tr>
<tr style="border-bottom: 1px solid gray">
<td colspan="2">

2. Install Pytorch and dependencies 

```pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html -f https://pytorch-geometric.com/whl/torch-1.5.0.html```

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

```pip install faiss-cpu```
    
</td>
<td style="border-left: 1px solid gray">

4. Install GPU-optimized packages

```pip install faiss-gpu cupy-cudaXXX```, with ```XXX```
    
</td>
</tr>
</table>

