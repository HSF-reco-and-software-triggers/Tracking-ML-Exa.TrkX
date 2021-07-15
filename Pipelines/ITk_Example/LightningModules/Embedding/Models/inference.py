import sys
import os
import copy
import logging
import tracemalloc
import gc
from memory_profiler import profile

from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
import numpy as np

from ..utils import fetch_pt, build_edges, graph_intersection

"""
Class-based Callback inference for integration with Lightning
"""


class EmbeddingTelemetry(Callback):

    """
    This callback contains standardised tests of the performance of a GNN
    """

    def __init__(self):
        super().__init__()
        logging.info("Constructing telemetry callback")

    def on_test_start(self, trainer, pl_module):

        """
        This hook is automatically called when the model is tested after training. The best checkpoint is automatically loaded
        """
        self.preds = []
        self.truth = []
        self.truth_graph = []
        self.pt_true_pos = []
        self.pt_true = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        """
        Get the relevant outputs from each batch
        """
        pts = fetch_pt(batch)
        true_positives = outputs["preds"][:, outputs["truth"]]
        true = outputs["truth_graph"]

        self.pt_true_pos.append(pts[true_positives])
        self.pt_true.append(pts[true])

        print(pts.shape, true_positives.shape, true.shape)

    #         self.preds.append(outputs["preds"])
    #         self.truth.append(outputs["truth"])
    #         self.truth_graph.append(outputs["truth_graph"])

    def on_test_end(self, trainer, pl_module):

        """
        1. Aggregate all outputs,
        2. Calculate the ROC curve,
        3. Plot ROC curve,
        4. Save plots to PDF 'metrics.pdf'
        """

        # REFACTOR THIS INTO CALCULATE METRICS, PLOT METRICS, SAVE METRICS
        pt_true_pos = np.concatenate(self.pt_true_pos, axis=1)
        pt_true = np.concatenate(self.pt_true, axis=1)

        print(pt_true_pos.shape, pt_true.shape)

        pt_true_pos_av = (pt_true_pos[0] + pt_true_pos[1]) / 2
        pt_true_av = (pt_true[0] + pt_true[1]) / 2

        #         bins = np.arange(pl_module.hparams["pt_min"], np.ceil(pt_true_av.max()), 0.5)
        #         bins = np.logspace(np.log(np.floor(pt_true_av.min())), np.log(np.ceil(pt_true_av.max())), 10)
        bins = np.logspace(0, 1.5, 10)
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        tp_hist = np.histogram(pt_true_pos_av, bins=bins)[0]
        t_hist = np.histogram(pt_true_av, bins=bins)[0]
        ratio_hist = tp_hist / t_hist

        # Update this to dynamically adapt to number of metrics
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        axs = axs.flatten() if type(axs) is list else [axs]

        axs[0].plot(centers, ratio_hist)
        axs[0].plot([centers[0], centers[-1]], [1, 1], "--")
        axs[0].set_xlabel("pT (GeV)")
        axs[0].set_ylabel("Efficiency")
        axs[0].set_title("Metric Learning Efficiency")
        plt.tight_layout()

        os.makedirs(pl_module.hparams.output_dir, exist_ok=True)
        fig.savefig(
            os.path.join(pl_module.hparams.output_dir, "metrics.pdf"), format="pdf"
        )


class EmbeddingInferenceCallback(Callback):
    def __init__(self):
        self.output_dir = None
        self.overwrite = False

    def on_train_start(self, trainer, pl_module):
        # Prep the directory to produce inference data to
        self.output_dir = pl_module.hparams.output_dir
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

        # Set overwrite setting if it is in config
        self.overwrite = (
            pl_module.hparams.overwrite if "overwrite" in pl_module.hparams else False
        )

    def on_train_end(self, trainer, pl_module):
        print("Training finished, running inference to build graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": pl_module.trainset,
            "val": pl_module.valset,
            "test": pl_module.testset,
        }
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0
        pl_module.eval()
        tracemalloc.start()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for batch_idx, batch in enumerate(dataset):
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
                            pl_module.device
                        )  # Is this step necessary??
                        self.construct_downstream(batch_to_save, pl_module, datatype)

                    batch_incr += 1

    def construct_downstream(self, batch, pl_module, datatype):

        # Free up batch.weights for subset of embedding selection
        batch.true_weights = batch.weights

        if "ci" in pl_module.hparams["regime"]:
            spatial = pl_module(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = pl_module(batch.x)

        # Make truth bidirectional
        e_bidir = torch.cat(
            [
                batch.layerless_true_edges,
                torch.stack(
                    [batch.layerless_true_edges[1], batch.layerless_true_edges[0]],
                    axis=1,
                ).T,
            ],
            axis=-1,
        )

        # Build the radius graph with radius < r_test
        e_spatial = build_edges(
            spatial, pl_module.hparams.r_test, 300
        )  # This step should remove reliance on r_val, and instead compute an r_build based on the EXACT r required to reach target eff/pur

        # Arbitrary ordering to remove half of the duplicate edges
        R_dist = torch.sqrt(batch.x[:, 0] ** 2 + batch.x[:, 2] ** 2)
        e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]

        if "weighting" in pl_module.hparams["regime"]:
            weights_bidir = torch.cat([batch.weights, batch.weights])
            e_spatial, y_cluster, new_weights = graph_intersection(
                e_spatial, e_bidir, using_weights=True, weights_bidir=weights_bidir
            )
            batch.weights = new_weights
        else:
            e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        # Re-introduce random direction, to avoid training bias
        random_flip = torch.randint(2, (e_spatial.shape[1],)).bool()
        e_spatial[0, random_flip], e_spatial[1, random_flip] = (
            e_spatial[1, random_flip],
            e_spatial[0, random_flip],
        )

        batch.edge_index = e_spatial
        batch.y = y_cluster

        self.save_downstream(batch, pl_module, datatype)

    def save_downstream(self, batch, pl_module, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)

        logging.info("Saved event {}".format(batch.event_file[-4:]))
