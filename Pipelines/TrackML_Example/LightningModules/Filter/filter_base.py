# System imports
import sys, os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import numpy as np
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import graph_intersection, load_dataset


class FilterBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering
        input_dirs = [None, None, None]
        input_dirs[: len(self.hparams["datatype_names"])] = [
            os.path.join(self.hparams["input_dir"], datatype)
            for datatype in self.hparams["datatype_names"]
        ]
        if "trainset" not in self.__dict__.keys():
            self.trainset, self.valset, self.testset = [
                load_dataset(
                    input_dir,
                    self.hparams["datatype_split"][i],
                    self.hparams["pt_background_min"],
                    self.hparams["pt_signal_min"],
                    self.hparams["true_edges"],
                    self.hparams["noise"],
                )
                for i, input_dir in enumerate(input_dirs)
            ]

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        emb = (
            None if (self.hparams["emb_channels"] == 0) else batch.embedding
        )  # Does this work??

        if self.hparams["ratio"] != 0:
            num_true, num_false = batch.y.bool().sum(), (~batch.y.bool()).sum()
            fake_indices = torch.where(~batch.y.bool())[0][
                torch.randint(num_false, (num_true.item() * self.hparams["ratio"],))
            ]
            true_indices = torch.where(batch.y.bool())[0]
            combined_indices = torch.cat([true_indices, fake_indices])
            # Shuffle indices:
            combined_indices = combined_indices[torch.randperm(len(combined_indices))]
            positive_weight = (
                torch.tensor(self.hparams["weight"])
                if ("weight" in self.hparams)
                else torch.tensor(self.hparams["ratio"])
            )

        else:
            combined_indices = torch.range(batch.edge_index.shape[1])
            positive_weight = (
                torch.tensor(self.hparams["weight"])
                if ("weight" in self.hparams)
                else torch.tensor((~batch.y.bool()).sum() / batch.y.sum())
            )

        output = (
            self(
                torch.cat([batch.cell_data, batch.x], axis=-1),
                batch.edge_index[:, combined_indices],
                emb,
            ).squeeze()
            if ("ci" in self.hparams["regime"])
            else self(batch.x, batch.edge_index[:, combined_indices], emb).squeeze()
        )

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights[combined_indices]
            manual_weights[batch.y[combined_indices] == 0] = 1
        else:
            manual_weights = None

        if "pid" in self.hparams["regime"]:
            y_pid = (
                batch.pid[batch.edge_index[0, combined_indices]]
                == batch.pid[batch.edge_index[1, combined_indices]]
            )
            loss = F.binary_cross_entropy_with_logits(
                output, y_pid.float(), weight=manual_weights, pos_weight=positive_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                output,
                batch.y[combined_indices].float(),
                weight=manual_weights,
                pos_weight=weight,
            )

        self.log("train_loss", loss)

        return result

    def shared_evaluation(self, batch, batch_idx, log=False):

        emb = (
            None if (self.hparams["emb_channels"] == 0) else batch.embedding
        )  # Does this work??

        cut_list = []
        val_loss = torch.tensor(0)
        for j in range(self.hparams["n_chunks"]):
            subset_ind = torch.chunk(
                torch.arange(batch.edge_index.shape[1]), self.hparams["n_chunks"]
            )[j]
            if "ci" in self.hparams["regime"]:
                output = self(
                    torch.cat([batch.cell_data, batch.x], axis=-1),
                    batch.edge_index[:, subset_ind],
                    emb,
                ).squeeze()
            else:
                output = self(batch.x, batch.edge_index[:, subset_ind], emb).squeeze()
                
            cut = F.sigmoid(output) > self.hparams["filter_cut"]
            cut_list.append(cut)

            if "weighting" in self.hparams["regime"]:
                manual_weights = batch.weights[subset_ind]
                manual_weights[batch.y[subset_ind] == 0] = 1
            else:
                manual_weights = None

            if "pid" not in self.hparams["regime"]:
                val_loss = +F.binary_cross_entropy_with_logits(
                    output, batch.y[subset_ind].float(), weight=manual_weights
                )
            else:
                y_pid = (
                    batch.pid[batch.edge_index[0, subset_ind]]
                    == batch.pid[batch.edge_index[1, subset_ind]]
                )
                val_loss = +F.binary_cross_entropy_with_logits(
                    output, y_pid.float(), weight=manual_weights
                )

        cut_list = torch.cat(cut_list)

        # Edge filter performance
        edge_positive = cut_list.sum().float()
        if "pid" in self.hparams["regime"]:
            true_y = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        else:
            true_y = batch.y

        edge_true = true_y.sum()
        edge_true_positive = (true_y.bool() & cut_list).sum().float()

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "eff": torch.tensor(edge_true_positive / edge_true),
                    "pur": torch.tensor(edge_true_positive / edge_positive),
                    "val_loss": val_loss,
                    "current_lr": current_lr,
                }
            )
        return {"loss": val_loss, "preds": score_list, "truth": true_y}

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


class FilterBaseBalanced(FilterBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

    def training_step(self, batch, batch_idx):

        emb = (
            None if (self.hparams["emb_channels"] == 0) else batch.embedding
        )  # Does this work??

        if "subset" in self.hparams["regime"]:
            subset_mask = np.isin(
                batch.edge_index.cpu(), batch.layerless_true_edges.unique().cpu()
            ).any(0)
            batch.edge_index = batch.edge_index[:, subset_mask]
            batch.y = batch.y[subset_mask]

        with torch.no_grad():
            cut_list = []
            for j in range(self.hparams["n_chunks"]):
                subset_ind = torch.chunk(
                    torch.arange(batch.edge_index.shape[1]), self.hparams["n_chunks"]
                )[j]
                if "ci" in self.hparams["regime"]:
                    output = self(
                        torch.cat([batch.cell_data, batch.x], axis=-1),
                        batch.edge_index[:, subset_ind],
                        emb,
                    ).squeeze()
                else:
                    output = self(batch.x, batch.edge_index[:, subset_ind], emb).squeeze()
                    
                cut = F.sigmoid(output) > self.hparams["filter_cut"]
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
            combined_indices[torch.randperm(len(combined_indices))]
            weight = torch.tensor(self.hparams["weight"])

        if "ci" in self.hparams["regime"]:
            output = self(
                torch.cat([batch.cell_data, batch.x], axis=-1),
                batch.edge_index[:, combined_indices],
                emb,
            ).squeeze()
        else:
            output = self(batch.x, batch.edge_index[:, combined_indices], emb).squeeze()

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights[combined_indices]
            manual_weights[batch.y[combined_indices] == 0] = 1
        else:
            manual_weights = None

        if "pid" in self.hparams["regime"]:
            y_pid = (
                batch.pid[batch.edge_index[0, combined_indices]]
                == batch.pid[batch.edge_index[1, combined_indices]]
            )
            loss = F.binary_cross_entropy_with_logits(
                output, y_pid.float(), weight=manual_weights, pos_weight=weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                output,
                batch.y[combined_indices].float(),
                weight=manual_weights,
                pos_weight=weight,
            )

        self.log_dict({"train_loss": loss}, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        result = self.shared_evaluation(batch, batch_idx, log=True)

        return result

    def test_step(self, batch, batch_idx):

        result = self.shared_evaluation(batch, batch_idx, log=False)

        return result

    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """

        emb = (
            None if (self.hparams["emb_channels"] == 0) else batch.embedding
        )  # Does this work??

        score_list = []
        val_loss = torch.tensor(0).to(self.device)
        for j in range(self.hparams["n_chunks"]):
            subset_ind = torch.chunk(
                torch.arange(batch.edge_index.shape[1]), self.hparams["n_chunks"]
            )[j]
            output = (
                self(
                    torch.cat([batch.cell_data, batch.x], axis=-1),
                    batch.edge_index[:, subset_ind],
                    emb,
                ).squeeze()
                if ("ci" in self.hparams["regime"])
                else self(batch.x, batch.edge_index[:, subset_ind], emb).squeeze()
            )
            scores = torch.sigmoid(output)
            score_list.append(scores)

            if "weighting" in self.hparams["regime"]:
                manual_weights = batch.weights[subset_ind]
                manual_weights[batch.y[subset_ind] == 0] = 1
            else:
                manual_weights = None

            if "pid" not in self.hparams["regime"]:
                val_loss = val_loss + F.binary_cross_entropy_with_logits(
                    output, batch.y[subset_ind].float(), weight=manual_weights
                )
            else:
                y_pid = (
                    batch.pid[batch.edge_index[0, subset_ind]]
                    == batch.pid[batch.edge_index[1, subset_ind]]
                )
                val_loss = +F.binary_cross_entropy_with_logits(
                    output, y_pid.float(), weight=manual_weights
                )

        score_list = torch.cat(score_list)
        cut_list = score_list > self.hparams["filter_cut"]

        # Edge filter performance
        edge_positive = cut_list.sum().float()
        if "pid" in self.hparams["regime"]:
            true_y = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        else:
            true_y = batch.y

        edge_true = true_y.sum()
        edge_true_positive = (true_y.bool() & cut_list).sum().float()

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]

            self.log_dict(
                {
                    "eff": torch.tensor(edge_true_positive / edge_true),
                    "pur": torch.tensor(edge_true_positive / edge_positive),
                    "val_loss": val_loss,
                    "current_lr": current_lr,
                }
            )
        return {"loss": val_loss, "preds": score_list, "truth": true_y}
