import sys, os
import logging

from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
import numpy as np

"""
Class-based Callback tracking performance for integration with Lightning
"""

from track_utils import get_tracking_metrics


class GNNTrackingTelemetry(Callback):

    """
    This callback contains standardised tests of the performance of a GNN
    """

    def __init__(self):
        super().__init__()
        logging.info("CONSTRUCTING CALLBACK!")

    def setup(self, trainer, pl_module, stage=None):
        
        # IDEALLY: Load particle and truth dataframes and attach them to the data
        
        # Process particles and truth and merge, trim particles by provided selection config file
        
    def on_validation_start(self, trainer, pl_module):

        """
        This hook is automatically called when the model is tested after training. The best checkpoint is automatically loaded
        """
        self.paricles = []
        self.truth = []
        self.edge_scores = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        """
        Get the relevant outputs from each batch
        """

        self.preds.append(outputs["preds"])
        self.truth.append(outputs["truth"])

    def on_validation_epoch_end(self, trainer, pl_module):

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
