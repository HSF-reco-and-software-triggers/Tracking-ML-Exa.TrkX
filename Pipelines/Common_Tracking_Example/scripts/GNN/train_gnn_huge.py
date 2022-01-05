import sys
import argparse
import yaml
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

sys.path.append("../../")
from LightningModules.GNN.Models.checkpoint_pyramid import CheckpointedPyramid

import wandb

from pytorch_lightning.plugins import DDPPlugin, DDP2Plugin, DDPSpawnPlugin
from pytorch_lightning.overrides import LightningDistributedModule

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
        
        
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train_gnn.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='example_gnn.yaml')
    return parser.parse_args()

def main():
    print("Running main")
    print(time.ctime())
    
    args = parse_args()
        
    with open(args.config) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    print("Initialising model")
    print(time.ctime())
    model = CheckpointedPyramid(default_configs)
    # model.setup(stage="fit")
    
    logger = WandbLogger(project=default_configs["project"], group="InitialTest", save_dir=default_configs["artifacts"])
    
    trainer = Trainer(gpus=4, num_nodes=8, strategy=CustomDDPPlugin(find_unused_parameters=False), max_epochs=default_configs["max_epochs"], logger=logger)
    trainer.fit(model)
    
    
if __name__ == "__main__":
    
    main()