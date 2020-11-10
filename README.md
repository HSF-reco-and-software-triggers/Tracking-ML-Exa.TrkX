<div align="center">

# Tracking with ML
### Exa.TrkX Collaboration


[Documentation](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/)

![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/murnanedaniel/acee2761c6c03febc3331296514ff721/raw/test.json) ![ci](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/workflows/ci/badge.svg)


</div>

Welcome to repository and documentation for ML pipelines and techniques by the ExatrkX Collaboration. Here we present a set of templates, best practices and results gathered from significant trial and error, to speed up the development of others in the domain of machine learning for high energy physics. We focus on applications specific to detector physics, but many tools can be applied to other areas, and these are collected in an application-agnostic way in the [Tools](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/tools/overview/) section.

## Intro

To start as quickly as possible, clone the repository and follow the steps in [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart.md). This will get you generating toy tracking data and running inference immediately. Many of the choices of structure will be made clear there. If you already have a particle physics problem in mind, you can apply the [Template](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/choosingguide.md) that is most suitable to your use case.

Once up and running, you may want to consider more complex ML [Models](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/models/overview/). Many of these are built on other libraries (for example [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)).

<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/application_diagram_1.png"/>
  <figcaption>The repository through an application lens, using Pipelines for specific physics goals</figcaption>
</figure>
