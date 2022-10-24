# System imports
import os
import sys
import yaml

# External imports
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.metrics import auc
import numpy as np
import pandas as pd
# import seaborn as sns
from tqdm import tqdm
import copy

# import seaborn as sns
import torch
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GNNInferenceCallback():
    def __init__(self, model, overwrite=None):
        self.model = model
        self.output_dir = model.hparams["output_dir"]
        if overwrite is not None: 
            self.overwrite = overwrite
        elif "overwrite" in model.hparams:
            self.overwrite = model.hparams["overwrite"]
        else:
            self.overwrite = False

        # Prep the directory to produce inference data to
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

    def infer(self):
        print("Training finished, running inference to filter graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": self.model.trainset,
            "val": self.model.valset,
            "test": self.model.testset,
        }

        self.model.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                print(f"Building {datatype}")
                for batch in tqdm(dataset):
                    if (
                        not os.path.exists(
                            os.path.join(
                                self.output_dir, datatype, batch.event_file[-4:]
                            )
                        )
                    ) or self.overwrite:
                        batch = batch.to(self.model.device)
                        batch = self.construct_downstream(batch)
                        self.save_downstream(batch, datatype)


    def construct_downstream(self, batch):

        output = self.model.shared_evaluation(batch, 0, log=False)
        
        batch.scores = output["score"][: int(len(output["score"]) / 2)]

        return batch

    def save_downstream(self, batch, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)

def list_files(dir):
    """
    List files from directory dir 
    """
    files = os.listdir(dir)
    files = [os.path.join(dir, f) for f in files]
    files = sorted(files)
    return files

class FilterInferenceBuilder:
    def __init__(self, model, output_dir, hetero=True, overwrite=False):
        self.output_dir = output_dir
        self.model = model
        self.overwrite = overwrite
        self.hetero = hetero #Whether to use heterogeneous or homogeneous dataset

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
        self.model.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for event_idx, event_file in tqdm(enumerate(dataset)):
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

    def construct_downstream(self, batch, datatype):

        cut_list = []
        for j in range(self.model.hparams["n_chunks"]):
            subset_ind = torch.chunk(
                torch.arange(batch.edge_index.shape[1]), self.model.hparams["n_chunks"]
            )[j]
            if self.hetero:
                output = self.model(batch.x.float(), batch.cell_data[:, : self.model.hparams["cell_channels"]].float(), batch.edge_index[:, subset_ind], batch.volume_id).squeeze()
            else:
                output = (
                self.model(
                    torch.cat(
                        [
                            batch.cell_data[:, : self.model.hparams["cell_channels"]],
                            batch.x,
                        ],
                        axis=-1,
                    ),
                    batch.edge_index[:, subset_ind],
                ).squeeze()
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