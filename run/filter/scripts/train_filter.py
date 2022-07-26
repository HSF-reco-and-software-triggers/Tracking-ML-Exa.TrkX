"""
This script runs step 1 of the TrackML Quickstart example:
Training the metric learning model.
"""
import sys
import os
import argparse
import logging
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from utils import headline
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
sys.path.append("/global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX")
from Pipelines.TrackML_Example_Dev.LightningModules.Filter.Models.pyramid_filter import PyramidFilter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train_filter_network.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="configs/filter_config.yaml")
    add_arg('--load_ckpt', required=False)
    return parser.parse_args()


def train(config_file="configs/filter_config.yaml", load_ckpt=None):

    logging.info(headline("Running filter training"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    filter_configs = all_configs["filter_configs"]

    logging.info(headline("a) Initialising model"))

    model = PyramidFilter(filter_configs)
    logging.info(headline("b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"])
    os.makedirs(save_directory, exist_ok=True)
    logger = []
    for lg in filter_configs.get('loggers', []):
        if lg == 'CSVLogger':
            logger.append(CSVLogger(save_directory, name=common_configs["experiment_name"]))
        if lg == 'WandbLogger':
            logger.append(WandbLogger(project='TrackML', group='filter', save_dir=common_configs['wandb_save_dir']))

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        num_nodes=os.environ.get('SLURM_JOB_NUM_NODES') or 1,
        devices=common_configs["gpus"],
        max_epochs=filter_configs["max_epochs"],
        logger=logger
    )
    start = datetime.now()
    trainer.fit(model, ckpt_path=load_ckpt)
    logging.info(f"Training takes {datetime.now() - start}")

    logging.info(headline("c) Saving model") )

    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config
    load_ckpt = args.load_ckpt

    trainer, model = train(config_file, load_ckpt)
