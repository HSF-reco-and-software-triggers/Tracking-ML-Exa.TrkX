# Tracking ML Pipelines

- Define pipeline clearly
  - Pytorch lightning & MLFlow
  - Data processing
  - Graph construction
  - Graph neural network
  - Post-processing

Follow the tutorial in [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart) to get up and running quickly with a toy model.


## How do Pipelines work?

The aim of our pipeline structure is to abstract out as much repetitive code as possible. A pipeline is defined by a YAML config file, which only requires three inputs: The location of your model definitions `model_library`, the location to save/load your artifacts `artifact_library`, and the stages of the pipeline `stage_list`. An example stage list is
```yaml
- {set: GNN, name: SliceCheckpointedResAGNN, config: train_gnn.yaml}
```
The `set` key defines the type of ML model (to help avoid naming ambiguity), the `name` key is the class of the model, and the `config` key is the file specifying your choice of hyperparameters, directories, callbacks and so on. And that's it!

<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/pipeline_diagram_1.png"/>
  <figcaption>The repository through an application lens, using Pipelines for specific physics goals</figcaption>
</figure>

## Why this choice of abstraction?

I found that my two forms of R&D fell into breadth and depth. Much of the time, I would play at length with hyperparameters and model definitions, in which case I want that all to live in one place: The model's config file. Thus the pipeline config can remain untouched if we have one for each choice of (model, config) , or only changed occasionally if we choose to have only one. At other times, development would require a series of models, where successive results depend on hyperparameter choices earlier in the chain. Then I can play with the higher level pipeline config and try difference (model, config) stages, while the whole chain of hyperparameters is committed to each step via a logging platform (Weights & Biases in my case).


## Pytorch Lightning & MLFlow

This repository uses Pytorch Lightning, which allows us to encapsulate all training and model logic into a module object. This module is what is being specified by the pipeline config, with `name`. Combined with callbacks in the model config file, all pipeline logic is contained in each module. A callback object integrates with the module and knows about telemetry and post-processing steps. Rather than a monolithic function that passes data through the pipeline (whereby mistakes could be made), the pipeline asks each model how it should be treated, and then acts with the model's own methods.


## Data Processing & Modularity

**Todo**: Define clearly the data structure of each modular section
