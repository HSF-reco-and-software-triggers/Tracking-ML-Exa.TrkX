import sys
import argparse
import yaml
import time

import torch
import numpy
import random
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

sys.path.append("../../")
from LightningModules.GNN.Models.checkpoint_pyramid import CheckpointedPyramid
from LightningModules.GNN.Models.interaction_gnn import InteractionGNN
from LightningModules.GNN.Models.multi_interaction_gnn import MultiInteractionGNN
from LightningModules.GNN.Models.vanilla_checkagnn import VanillaCheckResAGNN

import wandb

from pytorch_lightning.plugins import DDPPlugin, DDP2Plugin, DDPSpawnPlugin
from pytorch_lightning.overrides import LightningDistributedModule

from pytorch_lightning import seed_everything


class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()
        
class CustomDDPSpawnPlugin(DDPSpawnPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()

class CustomDDP2Plugin(DDP2Plugin):
    def setup(self):
        # set the task idx
        self.task_idx = self.cluster_environment.local_rank()
        # self._model._set_static_graph()

def set_random_seed(seed):
    torch.random.manual_seed(seed)
    print("Random seed:", seed)
    seed_everything(seed)
    
        
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train_gnn.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='default_config.yaml')
    add_arg('random_seed', nargs='?', default=None)
    return parser.parse_args()

def main():
    print("Running main")
    print(time.ctime())
    
    args = parse_args()

    with open(args.config) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    # Set random seed
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
        default_configs["random_seed"] = args.random_seed
        
    elif "random_seed" in default_configs.keys():
        set_random_seed(default_configs["random_seed"])
    
    print("Initialising model")
    print(time.ctime())
    model_name = eval(default_configs["model"])
    model = model_name(default_configs)
    
    logger = WandbLogger(project=default_configs["project"], group="InitialTest", save_dir=default_configs["artifacts"])
    logger.watch(model, log="all")
    
    trainer = Trainer(gpus=4, num_nodes=2, max_epochs=default_configs["max_epochs"], logger=logger, strategy=CustomDDPPlugin(find_unused_parameters=False))
    trainer.fit(model)
    
    
if __name__ == "__main__":
    
    main()