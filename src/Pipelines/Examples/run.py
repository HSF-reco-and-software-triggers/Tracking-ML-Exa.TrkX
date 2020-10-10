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



# ================================== Embedding ==========================
    with open("LightningModules/Embedding/train_embedding.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = LayerlessEmbedding(config)
    callback_list = [EmbeddingInferenceCallback()]
    trainer = pl.Trainer(
        max_epochs = 1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=callback_list
    )
    trainer.fit(model)


# ================================== Filtering ==========================
    with open("LightningModules/Filter/train_filter.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = VanillaFilter(config)
    callback_list = [FilterInferenceCallback()]
    trainer = pl.Trainer(
        max_epochs = 1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=callback_list
    )
    trainer.fit(model)

if __name__=="main":
    main()
