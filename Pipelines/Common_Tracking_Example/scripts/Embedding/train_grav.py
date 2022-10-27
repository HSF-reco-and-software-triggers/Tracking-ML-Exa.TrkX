import sys
import argparse
import yaml
import time
import os
import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.overrides import LightningDistributedModule

sys.path.append("../../")
from LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from LightningModules.SuperEmbedding.Models.gravmetric import UndirectedGravMetric, DirectedGravMetric
from LightningModules.SuperEmbedding.Models.gravmetric2 import GravMetric
from LightningModules.SuperEmbedding.Models.agnn import ResAGNN
from LightningModules.SuperEmbedding.Models.grav_agnn import GravAGNN
from LightningModules.SuperEmbedding.Models.multi_grav_agnn import MultiGravAGNN
from LightningModules.SuperEmbedding.Models.undirected_embedding import UndirectedEmbedding
from LightningModules.SuperEmbedding.Models.directed_embedding import DirectedEmbedding

class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()
        
# ------------------------------- Convenience functions ----------------------
        
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="default_config.yaml")
    add_arg("root_dir", nargs="?", default=None)
    add_arg("checkpoint", nargs="?", default=None)
    return parser.parse_args()

def find_latest_checkpoint(checkpoint_path):
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.ckpt"))
    return max(checkpoint_files, key=os.path.getctime) if checkpoint_files else None

def get_default_root_dir(root_dir):
    if root_dir is None: 
        return os.path.join(".", os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
    else:
        return os.path.join(".", root_dir)

def load_config_and_checkpoint(config_path, default_root_dir):
    # Check if there is a checkpoint to load
    checkpoint = find_latest_checkpoint(default_root_dir) if default_root_dir is not None else None
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        return torch.load(checkpoint, map_location=torch.device("cpu"))["hyper_parameters"], checkpoint
    else:
        print("No checkpoint found, loading config from file")
        with open(config_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader), None

def get_model(configs, checkpoint):
    model_name = eval(configs["model"])
    if checkpoint:
        return model_name.load_from_checkpoint(checkpoint)
    else:
        return model_name(configs)

def get_trainer(configs, default_root_dir):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=2, save_last=True
    )

    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else None
    logger = WandbLogger(
        project=configs["project"],
        save_dir=configs["artifacts"],
        id=job_id,
    )

    return Trainer(
        gpus=configs["gpus"],
        max_epochs=configs["max_epochs"],
        logger=logger,
        strategy=CustomDDPPlugin(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        default_root_dir=default_root_dir
    )

# ------------------------------- Main ----------------------

def main():
    args = parse_args()
    default_root_dir = get_default_root_dir(args.root_dir)
    configs, checkpoint = load_config_and_checkpoint(args.config, default_root_dir)
    model = get_model(configs, checkpoint)
    trainer = get_trainer(configs, default_root_dir)
    trainer.fit(model)

if __name__ == "__main__":

    main()
