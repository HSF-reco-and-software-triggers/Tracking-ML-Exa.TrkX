import sys, os
import logging

from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
import numpy as np

from ..utils import fetch_pt

'''
Class-based Callback inference for integration with Lightning
'''

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

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

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
        
        pt_true_pos_av = (pt_true_pos[0] + pt_true_pos[1])/2
        pt_true_av = (pt_true[0] + pt_true[1])/2
        
#         bins = np.arange(pl_module.hparams["pt_min"], np.ceil(pt_true_av.max()), 0.5)
#         bins = np.logspace(np.log(np.floor(pt_true_av.min())), np.log(np.ceil(pt_true_av.max())), 10)
        bins = np.logspace(0, 1.5, 10)
        centers = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        
        tp_hist = np.histogram(pt_true_pos_av, bins=bins)[0]
        t_hist = np.histogram(pt_true_av, bins=bins)[0]
        ratio_hist = tp_hist / t_hist
        
        # Update this to dynamically adapt to number of metrics
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
        axs = axs.flatten() if type(axs) is list else [axs]

        axs[0].plot(centers, ratio_hist)
        axs[0].plot([centers[0], centers[-1]], [1, 1], '--')
        axs[0].set_xlabel('pT (GeV)')
        axs[0].set_ylabel('Efficiency')
        axs[0].set_title('Metric Learning Efficiency')
        plt.tight_layout()
        
        os.makedirs(pl_module.hparams.output_dir, exist_ok=True)
        fig.savefig(os.path.join(pl_module.hparams.output_dir, "metrics.pdf"), format="pdf")
