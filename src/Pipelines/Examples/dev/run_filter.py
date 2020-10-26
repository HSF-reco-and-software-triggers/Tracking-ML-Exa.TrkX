import torch
import pytorch_lightning as pl
import yaml
import importlib
from LightningModules.Embedding.layerless_embedding import LayerlessEmbedding, EmbeddingInferenceCallback
from LightningModules.Embedding.utils import get_best_run, build_edges, res, graph_intersection
from LightningModules.Filter.utils import stringlist_to_classes
from LightningModules.Filter.vanilla_filter import VanillaFilter, FilterInferenceCallback
from LightningModules.Processing.feature_construction import FeatureStore
from pytorch_lightning.loggers import WandbLogger


def main():


# ================================== Filtering ==========================
    with open("LightningModules/Filter/train_filter.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("Training with:", config)
        
    model = VanillaFilter(config)
    wandb_logger = WandbLogger(project="FilteringStudy", group="LayerlessEndcaps", log_model=True, save_dir = config["wandb_save_dir"])
    trainer = pl.Trainer(
        max_epochs = config['max_epochs'],
        logger=wandb_logger,
        gpus=1,
        callbacks=stringlist_to_classes(config["callbacks"])
    )
        
    trainer.fit(model)

if __name__=="__main__":
    main()
