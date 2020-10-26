import torch
import pytorch_lightning as pl
import yaml
import importlib
from LightningModules.Embedding.layerless_embedding import LayerlessEmbedding, EmbeddingInferenceCallback
from LightningModules.Filter.vanilla_filter import VanillaFilter, FilterInferenceCallback
from LightningModules.Processing.feature_construction import FeatureStore


def main():

# ================================== Preprocessing ==========================
    with open("LightningModules/Processing/prepare_feature_store.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    preprocess_dm = FeatureStore(config)
    preprocess_dm.prepare_data()


if __name__=="__main__":
    main()
