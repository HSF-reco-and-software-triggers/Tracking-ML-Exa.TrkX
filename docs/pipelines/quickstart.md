# Quickstart Tutorial

## 1. Install

See instructions at [Install](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/#install).

## 2. Get Dataset

For now, let's keep our data in a directory, with a location saved to `/my/data/path`: 
```
export EXATRKX_DATA=/my/data/path
```
(you can hard-code these into your custom configs later).

The easiest way to get the TrackML dataset is to use the Kaggle API. Install it with
```
pip install kaggle
```
and grab a small toy dataset with
```
kaggle competitions download \
    -c trackml-particle-identification \
    -f train_sample.zip \
    -p $EXATRKX_DATA
```

## 3. Running the Pipeline

### Configuration

A pipeline runs at three layers of configuration, to allow for as much flexibility as possible. To get running immediately however, you needn't change any of the defaults. From the `Pipelines/TrackML_Example/` directory, we run

```
traintrack
```
which by default loads the pipeline specified in `configs/pipeline_test`. 

While it's running, get a cup of tea and a Tim-Tam, and let's see what it's doing:

### Default behaviour

Our quickstart pipeline is running **three** stages, with a single configuration for each. You can see in `config/pipeline_test.yaml` that the three stages are:

- A **Processing** stage with the class `FeatureStore` and config `prepare_small_feature_store.yaml`;
- An **Embedding** stage with the class `LayerlessEmbedding` and config `train_small_embedding.yaml`; and
- A **Filter** stage with the class `VanillaFilter` and config `train_small_filter.yaml`.

The **Processing** stage is exactly that: data processing. It is not "trainable", and so the pipeline treats it differently than a trainable stage. Under the hood, it is a LightningDataModule, rather than the trainable models, which inherit from LightningModule. In this case, `FeatureStore` is performing some calculations on the cell information in the detector, and constructing truth graphs that will later be used for training. These calculations are computationally expensive, so it doesn't make sense to calculate them on-the-fly while training. 

The trainable models **Embedding** and **Filter** are learning the non-linear metric of the truth graphs, and pairwise likelihoods of hits sharing a truth graph edge, respectively. The details are not so important at this stage, what matters is that these stages are modular: Each one can be run alone, but by adding a **callback** to the end, it can prepare the dataset for the next stage. Looking at `LightningModules/Embedding/train_small_embedding.yaml` you will see that a callback is given as `callbacks: EmbeddingInferenceCallback`. Any number of callbacks can be added, and they adhere to the [Lightning callback system](https://pytorch-lightning.readthedocs.io/en/latest/callbacks.html). The one referred to here runs the best version of the trained Embedding model on each directory of the data split (train, val, test) and saves it for the next stage. We could also add a telemetry callback, e.g. the `EmbeddingTelemetry` callback in `LightningModules/Embedding/Models/inference.py`. This callback prints a PDF of the transverse momentum vs. the efficiency of the metric learning model, saving it in the `output_dir`. It "hooks" into the testing phase, which is run after every training phase:



The default settings of this run pull from the three configuration files given at each stage. You can look at them


- Global config:
  - Model library: This is in the repository
  - Artifact library: This should be a place where you can store data = checkpoints
- Pipeline config: explain stages in it
- Explain default model config parameters (1GeV, n_train=90, n_val/n_test = 5)
