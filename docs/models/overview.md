# Models for Tracking ML

## Including a New Model

This repository aims to collect modular ML models that can be reused for various HEP applications. As such, new models are welcome to be included. Currently, new models can either be added to the general model collection in `src/Architectures/` or, if the model is intended to be used in an example application, in that application's pipeline in `src/Pipelines/APPLICATION/LightningModules/`. Models are organised by their architecture, which allows much of the common code to be abstracted out. For example, a particular convolution of a GNN can be specified in `src/.../GNN/Models/my_new_gnn.py` which inherits all the training behavior of `GNNBase` in `GNN/gnn_base.py`. 

The models are written in Pytorch Lightning, to avoid as much training boilerplate as possible. To include a new model, or a new base class that doesn't yet exist, it should be implemented in Pytorch Lightning. **It usually takes 5-10 minutes to convert vanilla Pytorch code to Pytorch Lightning**, see for example [this Lightning Tutorial.](https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html)

Once a model is included in a Pipeline, it can included as a stage, as in the [TrackML Example](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/master/src/Pipelines/TrackML_Example/configs/pipeline_quickstart.yaml).

## Available Models

- Embeddings
- Edge MLPs
- Static GNNs
- Dynamic GNNs

## Top Takeaways

- Best model for each use case (by memory, timing, etc.)
- Best hyperparameter choice

## Test Model

<!-- ::: src.Pipelines.TrackML_Example.LightningModules.Embedding.embedding_base.EmbeddingBase
    handler: python
    selection:
      members:
        - training_step
        - validation_step
    rendering:
      show_root_heading: true
      show_source: true -->

<!-- ::: src.Pipelines.Examples.LightningModules.GNN.gnn_base.GNNBase -->

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
