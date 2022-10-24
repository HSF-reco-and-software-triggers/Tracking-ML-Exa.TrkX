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
        
    def balance_volume_loss(self, output, y, volume_id, edge_index, weight):
        """
        Balance the true and fake samples in each volume-volume combination
        """

        # 1. Convert edge volume IDs to single volume label:
        volume_labels = self.get_volume_labels(volume_id, edge_index)
        
        # 2. Count labels in each true and fake set:
        true_labels = volume_labels[y.bool()]
        fake_labels = volume_labels[~y.bool()]
        true_labels, true_counts = torch.unique(true_labels, return_counts=True)
        fake_labels, fake_counts = torch.unique(fake_labels, return_counts=True)
        
        # 3. Make true and fake mappings:
        true_mapping = torch.zeros(10, dtype=torch.long, device=output.device)
        true_mapping[true_labels] = true_counts
        fake_mapping = torch.zeros(10, dtype=torch.long, device=output.device)
        fake_mapping[fake_labels] = fake_counts

        # 4. Calculate ratio of true and fake samples in each volume-volume combination:
        ft_ratio = fake_mapping / true_mapping
        
        # 5. Fix nan values:
        ft_ratio[torch.isnan(ft_ratio)] = 1

        # 6. Map volume-volume combinations to their ratio:
        volume_ft_ratio = ft_ratio[volume_labels]
        
        # 7. Ratio weight
        ratio_weight = torch.ones_like(output)
        ratio_weight[y.bool()] = volume_ft_ratio[y.bool()] * weight
        
        # 8. Calculate loss:
        loss = F.binary_cross_entropy_with_logits(
            output,
            y.float(),
            weight=ratio_weight,
        )

        return loss 

    def get_volume_labels(self, volume_id, edge_index):
        edge_vol_ids = volume_id[edge_index]
        volume_labels = self.vol_matrix[edge_vol_ids[0], edge_vol_ids[1]]
        return volume_labels.long().to(edge_index.device)

    def get_multi_vol_metrics(self, scores, truth, volume_id, edge_index):
        volume_labels = self.get_volume_labels(volume_id, edge_index)
        multi_vol_dict = {}
        for i in range(self.vol_matrix.max().int().item()+1):
            vol_idx = (volume_labels == i)
            vol_truth = truth[vol_idx]
            vol_scores = scores[vol_idx]
            if len(torch.unique(vol_truth)) > 1:
                vol_auc = roc_auc_score(vol_truth.cpu().detach(), vol_scores.cpu().detach())
            else:
                vol_auc = 0
            multi_vol_dict[f"multi_vol_metrics/auc_vol_{i}"] = vol_auc

        return multi_vol_dict          
        
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
            if "balance_volumes" in self.hparams.keys() and self.hparams["balance_volumes"]:
                loss = self.balance_volume_loss(output, batch.y[combined_indices].float(), batch.volume_id, batch.edge_index[:, combined_indices], weight)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    output, 
                    batch.y[combined_indices].float(), 
                    pos_weight=weight
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
            dict_to_log = {
                        "eff": eff,
                        "pur": pur,
                        "val_loss": val_loss,
                        "current_lr": current_lr,
                        "auc": auc,
                    }
            if "multi_vol_metrics" in self.hparams["regime"]:
                multi_vol_metrics = self.get_multi_vol_metrics(score_list, truth, batch.volume_id, batch.edge_index)
                dict_to_log.update(multi_vol_metrics)

            self.log_dict(dict_to_log)                 
            
            
        # print("Returning validation loss on device", self.device, )
        return {"loss": val_loss, "preds": score_list, "truth": truth}
