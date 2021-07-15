import sys, os
import logging

from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
import numpy as np

"""
Class-based Callback inference for integration with Lightning
"""


class GNNInferenceCallback(Callback):
    def __init__(self):
        self.output_dir = None
        self.overwrite = False

    def on_test_start(self, trainer, pl_module):
        # Prep the directory to produce inference data to
        self.output_dir = pl_module.hparams.output_dir
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

    def on_test_end(self, trainer, pl_module):
        print("Training finished, running inference to filter graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": pl_module.trainset,
            "val": pl_module.valset,
            "test": pl_module.testset,
        }
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0

        pl_module.eval()
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
                        batch = batch.to(pl_module.device)  # Is this step necessary??
                        batch = self.construct_downstream(batch, pl_module)
                        self.save_downstream(batch, pl_module, datatype)

                    batch_incr += 1

    def construct_downstream(self, batch, pl_module):

        emb = (
            None if (pl_module.hparams["emb_channels"] == 0) else batch.embedding
        )  # Does this work??

        output = (
            pl_module(
                torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius, emb
            ).squeeze()
            if ("ci" in pl_module.hparams["regime"])
            else pl_module(batch.x, batch.e_radius, emb).squeeze()
        )
        y_pid = batch.pid[batch.e_radius[0]] == batch.pid[batch.e_radius[1]]

        cut_indices = F.sigmoid(output) > pl_module.hparams["filter_cut"]
        batch.e_radius = batch.e_radius[:, cut_indices]
        batch.y_pid = y_pid[cut_indices]
        batch.y = batch.y[cut_indices]

        return batch

    def save_downstream(self, batch, pl_module, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)


class GNNTelemetry(Callback):

    """
    This callback contains standardised tests of the performance of a GNN
    """

    def __init__(self):
        super().__init__()
        logging.info("CONSTRUCTING CALLBACK!")

    def on_test_start(self, trainer, pl_module):

        """
        This hook is automatically called when the model is tested after training. The best checkpoint is automatically loaded
        """
        self.preds = []
        self.truth = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        """
        Get the relevant outputs from each batch
        """

        self.preds.append(outputs["preds"])
        self.truth.append(outputs["truth"])

    def on_test_end(self, trainer, pl_module):

        """
        1. Aggregate all outputs,
        2. Calculate the ROC curve,
        3. Plot ROC curve,
        4. Save plots to PDF 'metrics.pdf'
        """

        # REFACTOR THIS INTO CALCULATE METRICS, PLOT METRICS, SAVE METRICS
        preds = np.concatenate(self.preds)
        truth = np.concatenate(self.truth)
        print(preds.shape, truth.shape)

        roc_fpr, roc_tpr, roc_thresholds = sklearn.metrics.roc_curve(truth, preds)
        roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)
        logging.info("ROC AUC: %s", roc_auc)

        # Update this to dynamically adapt to number of metrics
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        axs = axs.flatten() if type(axs) is list else [axs]

        axs[0].plot(roc_fpr, roc_tpr)
        axs[0].plot([0, 1], [0, 1], "--")
        axs[0].set_xlabel("False positive rate")
        axs[0].set_ylabel("True positive rate")
        axs[0].set_title("ROC curve, AUC = %.3f" % roc_auc)
        plt.tight_layout()

        fig.savefig("metrics.pdf", format="pdf")
