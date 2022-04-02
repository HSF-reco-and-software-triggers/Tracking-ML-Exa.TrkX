# Models for Tracking ML

This repository aims to collect modular ML models that can be reused for various HEP applications. However, it is generally directed towards the Exatrkx pipeline for track reconstruction, which **handles HEP data as graph-structured**. This means that many of the models are related to either:

1. Constructing graphs (e.g. metric learning, filtering)
2. Learning node or edge feature representations (e.g. edge classification)
3. Segmenting and processing graphs (e.g. segment stitching, splitting)

If you have, for example, image-structured data, the pipeline management tool [TrainTrack](https://github.com/murnanedaniel/train-track) will still be useful, but many of these models will not. On the other hand, you might consider if your images are representations of some sparse underlying physics whether they could be better-represented as graphs...

## Including a New Model

 As such, new models are welcome to be included. Currently, new models can either be added to the general model collection in `Architectures` or, if the model is intended to be used in an example application, in that application's pipeline in `Pipelines/APPLICATION/LightningModules/`. Models are organised by their architecture, which allows much of the common code to be abstracted out. For example, a particular convolution of a GNN can be specified in `GNN/Models/my_new_gnn.py` which inherits all the training behavior of `GNNBase` in `GNN/gnn_base.py`. 

The models are written in `Pytorch Lightning`, to avoid as much training boilerplate as possible. To include a new model, or a new base class that doesn't yet exist, it should be implemented in Pytorch Lightning. **It usually takes 5-10 minutes to convert vanilla Pytorch code to Pytorch Lightning**, see for example [this Lightning Tutorial.](https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html)

Once a model is included in a Pipeline, it can included as a stage, as in the [TrackML Example](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/master/src/Pipelines/TrackML_Example/configs/pipeline_quickstart.yaml).

## Available Models

### Pre-Processing

In order to easily consume data for machine learning, it is beneficial to package it into a "feature store". This takes the large, possibly messy, possibly text/csv-format, possibly missing dataset and makes it uniform and as light as it can be. The feature store processor here uses the Pytorch Geometric (PyG) `data` class, for several reasons: a) It is compatible with the PyG `dataloader` class, which takes it as a simple list and we therefore seriously reduce boilerplate; b) It is a class than can be entirely placed on and off a device with a simple `.to(device)`; and c) The PyG `dataloader` automatically batches graph-structured data in the correct way.

The `FeatureStore` class is a `LightningDataModule` that is compatible with the `TrainTrack` system - that is, it doesn't do any training or inference, but is instead understood to be a processing stage. It does this by implementing a `prepare_data` method, and the rest is automatic.

In the TrackML example, the most important tasks of `FeatureStore` processing are to select node (i.e. spacepoint) features and normalise them, and define a truth graph. The rest is housekeeping.

### Embedding Models (i.e. Metric Learning)


### Filtering Models


### Graph Neural Networks


### Segmentation




## Top Takeaways

- Best model for each use case (by memory, timing, etc.)
- Best hyperparameter choice

## Test Model

<!-- ::: Pipelines.TrackML_Example.LightningModules.Embedding.embedding_base.EmbeddingBase
    handler: python -->
<!--     selection:
      members:
        - training_step
        - validation_step
    rendering:
      show_root_heading: true
      show_source: true -->

<!-- ::: gnn_base.GNNBase
    handler: python -->

<!-- ::: src.Pipelines.Examples.LightningModules.GNN.Models.interaction_gnn.InteractionGNN
    handler: python
    selection:
      members:
        - forward
    rendering:
      show_root_heading: false
      show_source: false

::: GNN.Models.interaction_gnn.InteractionGNN
    handler: python
    selection:
      members:
        - forward
    rendering:
      show_root_heading: false
      show_source: false -->
