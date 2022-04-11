# System imports
import sys, os

# 3rd party imports
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeoLoader
import numpy as np

from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import graph_intersection, load_dataset, filter_dataset, LargeDataset
from .filter_base import LargeFilterBaseBalanced

class HeteroFilterBase(LargeFilterBaseBalanced):
    def __init__(self, hparams):
        super().__init__(hparams)
        
    def training_step(self, batch, batch_idx):

        # Handle training towards a subset of the data
        if "subset" in self.hparams["regime"]:
            subset_mask = np.isin(
                batch.edge_index.cpu(), batch.signal_true_edges.unique().cpu()
            ).any(0)
            batch.edge_index = batch.edge_index[:, subset_mask]
            batch.y = batch.y[subset_mask]
            batch.y_pid = batch.y_pid[subset_mask]

        with torch.no_grad():
            cut_list = []
            for j in range(self.hparams["n_chunks"]):
                # print("Loading chunk", j, "on device", self.device)
                subset_ind = torch.chunk(
                    torch.arange(batch.edge_index.shape[1]), self.hparams["n_chunks"]
                )[j]
                output = self(batch.x.float(), batch.cell_data[:, :self.hparams["cell_channels"]].float(), batch.edge_index[:, subset_ind], batch.volume_id).squeeze()
                    
                cut = torch.sigmoid(output) > self.hparams["edge_cut"]
                cut_list.append(cut)

            cut_list = torch.cat(cut_list)

            num_true, num_false = batch.y.bool().sum(), (~batch.y.bool()).sum()
            true_indices = torch.where(batch.y.bool())[0]
            hard_negatives = cut_list & ~batch.y.bool()
            hard_indices = torch.where(hard_negatives)[0]
            hard_indices = hard_indices[torch.randperm(len(hard_indices))][
                : int(len(true_indices) * self.hparams["ratio"] / 2)
            ]
            easy_indices = torch.where(~batch.y.bool())[0][
                torch.randint(
                    num_false, (int(num_true.item() * self.hparams["ratio"] / 2),)
                )
            ]

            combined_indices = torch.cat([true_indices, hard_indices, easy_indices])

            # Shuffle indices:
            combined_indices = combined_indices[torch.randperm(len(combined_indices))][
                : self.hparams["edges_per_batch"]
            ]
        
        weight = torch.tensor(self.hparams["weight"])
        output = self(batch.x.float(), batch.cell_data[:, :self.hparams["cell_channels"]].float(), batch.edge_index[:, combined_indices], batch.volume_id).squeeze()

        if "pid" in self.hparams["regime"]:
            loss = F.binary_cross_entropy_with_logits(
                output, 
                batch.y_pid[combined_indices].float(), 
                pos_weight=weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                output,
                batch.y[combined_indices].float(),
                pos_weight=weight,
            )

        self.log("train_loss", loss)
        
        # print("Returning training loss on device", self.device)
        return loss

    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """

        score_list = []
        val_loss = torch.tensor(0).to(self.device)
        for j in range(self.hparams["n_chunks"]):
            # print("Loading chunk", j, "on device", self.device)
            
            subset_ind = torch.chunk(
                torch.arange(batch.edge_index.shape[1]), self.hparams["n_chunks"]
            )[j]

            output = self(batch.x.float(), batch.cell_data[:, :self.hparams["cell_channels"]].float(), batch.edge_index[:, subset_ind], batch.volume_id).squeeze()

            scores = torch.sigmoid(output)
            score_list.append(scores)

            if "pid" not in self.hparams["regime"]:
                val_loss = val_loss + F.binary_cross_entropy_with_logits(
                    output, batch.y[subset_ind].float()
                )
            else:
                val_loss = +F.binary_cross_entropy_with_logits(
                    output, batch.y_pid[subset_ind].float()
                )

        score_list = torch.cat(score_list)

        # Edge filter performance
        if "pid" in self.hparams["regime"]:
            truth = batch.y_pid.bool()
        else:
            truth = batch.y.bool()

        eff, pur, auc = self.get_metrics(truth, score_list)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "eff": eff,
                    "pur": pur,
                    "val_loss": val_loss,
                    "current_lr": current_lr,
                    "auc": auc,
                },
                # sync_dist=True,
                # on_epoch=True
            )
            
        # print("Returning validation loss on device", self.device, )
        return {"loss": val_loss, "preds": score_list, "truth": truth}