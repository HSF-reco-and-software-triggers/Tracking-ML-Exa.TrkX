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
import seaborn as sns
import torch
import torch.functional as F
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
from sklearn.metrics import precision_recall_curve
import random
import copy

sys.path.append(os.environ['EXATRKX_WD'])

from Pipelines.Common_Tracking_Example.LightningModules.Embedding.utils import build_edges, graph_intersection

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingInferenceBuilder:
    def __init__(self, model, split, output_dir, overwrite=False):
        self.output_dir = output_dir
        self.model = model
        self.overwrite = overwrite
        self.split=split

        # Prep the directory to produce inference data to
        self.datatypes = ["train", "val", "test"]

    def prepare_datastructure(self):
        # Prep the directory to produce inference data to
        self.output_dir = self.model.hparams.output_dir
        self.datatypes = ["train", "val", "test"]

        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

        all_events = os.listdir(self.model.hparams["input_dir"])
        random.shuffle(all_events)
        self.dataset_list = np.split(np.array(all_events), np.cumsum(self.split))
        
        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": list(self.dataset_list[0]),
            "val": list(self.dataset_list[1]),
            "test": list(self.dataset_list[2]),
        }
        
        return datasets

    def build(self):
        print("Training finished, running inference to build graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = self.prepare_datastructure()
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0

        self.model.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for batch_idx, batch_file in enumerate(dataset):
                    batch_file = os.path.join(self.model.hparams['input_dir'], batch_file)
                    batch = torch.load(batch_file)
                    percent = (batch_incr / total_length) * 100
                    sys.stdout.flush()
                    sys.stdout.write(f"{percent:.01f}% inference complete \r")
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

        if "ci" in self.model.hparams["regime"]:
            input_data = torch.cat(
                [batch.cell_data[:, : self.model.hparams["cell_channels"]], batch.x],
                axis=-1,
            )
            input_data[input_data != input_data] = 0
            spatial = self.model(input_data)
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0
            spatial = self.model(input_data)

        # Make truth bidirectional
        e_bidir = torch.cat(
            [
                batch.modulewise_true_edges,
                torch.stack(
                    [batch.modulewise_true_edges[1], batch.modulewise_true_edges[0]],
                    axis=1,
                ).T,
            ],
            axis=-1,
        )

        # Build the radius graph with radius < r_test
        e_spatial = build_edges(
            spatial,
            spatial,
            indices=None,
            r_max=self.model.hparams["r_test"],
            k_max=1000,
        )  # This step should remove reliance on r_val, and instead compute an r_build based on the EXACT r required to reach target eff/pur

        # Arbitrary ordering to remove half of the duplicate edges
        R_dist = torch.sqrt(batch.x[:, 0] ** 2 + batch.x[:, 2] ** 2)
        e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        # Re-introduce random direction, to avoid training bias
        random_flip = torch.randint(2, (e_spatial.shape[1],)).bool()
        e_spatial[0, random_flip], e_spatial[1, random_flip] = (
            e_spatial[1, random_flip],
            e_spatial[0, random_flip],
        )

        batch.edge_index = e_spatial
        batch.y = y_cluster

        self.save_downstream(batch, datatype)

    def save_downstream(self, batch, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)


# def main():

#     checkpoint_path = "/global/cscratch1/sd/danieltm/ExaTrkX/itk_lightning_checkpoints/ITk_1GeV/wnhns4e7/checkpoints/last.ckpt"
#     checkpoint = torch.load(checkpoint_path)
#     from LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding

#     model = LayerlessEmbedding.load_from_checkpoint(checkpoint_path).to(device)
#     model.eval()

#     output_dir = "/project/projectdirs/m3443/data/ITk-upgrade/processed/embedding_processed/1_GeV_unweighted"
#     model.hparams["train_split"] = [2000, 20, 20]
#     model.hparams["r_test"] = 0.9

#     model.setup(stage="fit")
#     edge_builder = EmbeddingInferenceBuilder(model, output_dir, overwrite=True)

#     edge_builder.build()


if __name__ == "__main__":
    main()
