# System imports
import os
import sys
import yaml
import argparse

# External imports
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.functional as F
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
from sklearn.metrics import precision_recall_curve
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

import copy

sys.path.append("/global/cfs/cdirs/m3443/usr/pmtuan/Tracking-ML-Exa.TrkX")
from Pipelines.TrackML_Example_Dev.LightningModules.Filter.Models.pyramid_filter import PyramidFilter
from utils import headline
# from LightningModules.Filter.Models.pyramid_filter import PyramidFilter

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="configs/filter_config.yaml")
    add_arg("--load_ckpt", required=False)
    return parser.parse_args()

class FilterInferenceBuilder:
    def __init__(self, model, output_dir, overwrite=False):
        self.output_dir = output_dir
        self.model = model
        self.overwrite = overwrite

        # Prep the directory to produce inference data to
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

        # Get [train, val, test] lists of files
        self.dataset_list = []
        for dataname in model.hparams["datatype_names"]:
            dataset = os.listdir(os.path.join(model.hparams["input_dir"], dataname))
            dataset = sorted(
                [
                    os.path.join(model.hparams["input_dir"], dataname, event)
                    for event in dataset
                ]
            )
            self.dataset_list.append(dataset)

    def build(self):
        print("Training finished, running inference to build graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": self.dataset_list[0],
            "val": self.dataset_list[1],
            "test": self.dataset_list[2],
        }
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0
        self.model.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for event_idx, event_file in enumerate(dataset):
                    percent = (batch_incr / total_length) * 100
                    sys.stdout.flush()
                    sys.stdout.write(f"{percent:.01f}% inference complete \r")
                    batch = torch.load(event_file).to(device)
                    if (
                        not os.path.exists(
                            os.path.join(
                                self.output_dir, datatype, batch.event_file[-4:]
                            )
                        )
                    ) or self.overwrite:
                        batch_to_save = copy.deepcopy(batch)
                        batch_to_save = batch_to_save.to(
                            self.model.device
                        )  # Is this step necessary??
                        self.construct_downstream(batch_to_save, datatype)

                    batch_incr += 1

    def construct_downstream(self, batch, datatype):

        emb = (
            None if (self.model.hparams["emb_channels"] == 0) else batch.embedding
        )  # Does this work??

        cut_list = []
        for j in range(self.model.hparams["n_chunks"]):
            subset_ind = torch.chunk(
                torch.arange(batch.edge_index.shape[1]), self.model.hparams["n_chunks"]
            )[j]
            output = (
                self.model(
                    torch.cat([batch.cell_data, batch.x], axis=-1),
                    batch.edge_index[:, subset_ind],
                    # emb,
                ).squeeze()
                if ("ci" in self.model.hparams["regime"])
                else self.model(batch.x, batch.edge_index[:, subset_ind], emb).squeeze()
            )
            cut = torch.sigmoid(output) > self.model.hparams["filter_cut"]
            cut_list.append(cut)

        cut_list = torch.cat(cut_list)

        if "pid" not in self.model.hparams["regime"]:
            batch.y = batch.y[cut_list]

        y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        batch.y_pid = y_pid[cut_list]
        batch.edge_index = batch.edge_index[:, cut_list]
        if "weighting" in self.model.hparams["regime"]:
            batch.weights = batch.weights[cut_list]

        self.save_downstream(batch, datatype)

    def save_downstream(self, batch, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)

def train(config_file="pipeline_config.yaml", load_ckpt=None):

    logging.info(headline("Step 4: Building graphs from filter model "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    filter_configs = all_configs["filter_configs"]

    logging.info(headline("a) Loading trained model" ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_ckpt = os.path.join(common_configs["artifact_directory"], common_configs["experiment_name"]+".ckpt")
    if load_ckpt is not None:
        logging.info(f"Loading checkpoint from {load_ckpt}")
        try:
            model = PyramidFilter.load_from_checkpoint(load_ckpt, emb_channels=0).to(device)
        except Exception as e:
            logging.warning(f'Unable to load checkpoint due to this exception:\n{e}\nRestoring from default checkpoint at {default_ckpt}')
            model = PyramidFilter.load_from_checkpoint(default_ckpt).to(device)
    else:
        logging.info(f"Loading checkpoint from {default_ckpt}")
        model = PyramidFilter.load_from_checkpoint(default_ckpt).to(device)

    logging.info(headline("b) Running inferencing"))
    split = (np.array(filter_configs['datatype_split']) / np.sum(filter_configs['datatype_split']) * filter_configs['n_inference_events']).astype(np.int16)
    split[0] = filter_configs['n_inference_events'] - np.sum(split[1:])
    model.hparams['n_events'] = filter_configs['n_inference_events']
    graph_builder = FilterInferenceBuilder(model, filter_configs['output_dir'])
    graph_builder.build()

    return graph_builder

def main():

    checkpoint_path = "/global/cscratch1/sd/danieltm/ExaTrkX//itk_lightning_checkpoints/ITk_1GeV_Filter/avcwv9al/checkpoints/last.ckpt"
    checkpoint = torch.load(checkpoint_path)

    model = PyramidFilter.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    output_dir = "/project/projectdirs/m3443/data/ITk-upgrade/processed/filter_processed/1_GeV_unweighted_high_eff"
    model.hparams["train_split"] = [2000, 20, 20]
    model.hparams["filter_cut"] = 0.15

    edge_builder = FilterInferenceBuilder(model, output_dir, overwrite=True)

    edge_builder.build()


if __name__ == "__main__":
    args = parse_args()

    graph_builder = train( args.config, args.load_ckpt )
