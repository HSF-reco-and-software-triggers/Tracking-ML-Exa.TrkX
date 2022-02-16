import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch
import numpy as np

from .utils import load_dataset, purity_sample
from sklearn.metrics import roc_auc_score


class GNNBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        # Assign hyperparameters
        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering

        if self.trainset is None:
            print("Setting up dataset")
            input_subdirs = [None, None, None]
            input_subdirs[: len(self.hparams["datatype_names"])] = [
                os.path.join(self.hparams["input_dir"], datatype)
                for datatype in self.hparams["datatype_names"]
            ]
            self.trainset, self.valset, self.testset = [
                load_dataset(
                    input_subdir=input_subdir,
                    num_events=self.hparams["datatype_split"][i],
                    **self.hparams
                )
                for i, input_subdir in enumerate(input_subdirs)
            ]

        if (
            (self.trainer)
            and ("logger" in self.trainer.__dict__.keys())
            and ("_experiment" in self.logger.__dict__.keys())
        ):
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.logger.experiment.define_metric("sig_auc", summary="max")
            self.logger.experiment.define_metric("tot_auc", summary="max")
            self.logger.experiment.define_metric("sig_fake_ratio", summary="max")
            self.logger.experiment.define_metric("custom_f1", summary="max")
            self.logger.experiment.log({"sig_auc": 0})
            self.logger.experiment.log({"sig_fake_ratio": 0})
            self.logger.experiment.log({"custom_f1": 0})

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=0
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=0
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=0
            )  # , pin_memory=True, persistent_workers=True)
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

    def get_input_data(self, batch):

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], axis=-1
            )
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0

        return input_data

    def handle_directed(self, batch, edge_sample, truth_sample):

        edge_sample = torch.cat([edge_sample, edge_sample.flip(0)], dim=-1)
        truth_sample = truth_sample.repeat(2)

        if ("directed" in self.hparams.keys()) and self.hparams["directed"]:
            direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
            edge_sample = edge_sample[:, direction_mask]
            truth_sample = truth_sample[direction_mask]

        return edge_sample, truth_sample

    def training_step(self, batch, batch_idx):

        truth = (
            batch.y_pid.bool() if "y_pid" in self.hparams["regime"] else batch.y.bool()
        )

        if ("train_purity" in self.hparams.keys()) and (
            self.hparams["train_purity"] > 0
        ):
            edge_sample, truth_sample = purity_sample(
                truth, self.hparams["train_purity"], self.hparams["regime"]
            )
        else:
            edge_sample, truth_sample = batch.edge_index, truth

        edge_sample, truth_sample = self.handle_directed(
            batch, edge_sample, truth_sample
        )

        # Handle training towards a subset of the data
        if "subset" in self.hparams["regime"]:
            subset_mask = np.isin(
                edge_sample.cpu(), batch.signal_true_edges.unique().cpu()
            ).any(0)
        else:
            subset_mask = torch.ones(edge_sample.shape[1]).bool()

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth_sample.bool()).sum() / truth_sample.sum())
        )

        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights[subset_mask]
        else:
            manual_weights = None

        if "y-y_pid" in self.hparams["regime"]:
            y_subset = (batch.y.bool() | ~batch.y_pid.bool()).repeat(2)
            output, truth_sample = output[y_subset], truth_sample[y_subset]

        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), weight=manual_weights, pos_weight=weight
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def log_metrics(self, output, batch, loss, log):

        preds = torch.sigmoid(output) > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        truth = batch.y.repeat(2)
        sig_true = truth.sum().float()
        sig_true_positive = (truth.bool() & preds).sum().float()
        sig_auc = roc_auc_score(
            truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Total true & total tp
        truth = (batch.y_pid.bool() | batch.y.bool()).repeat(2)
        tot_true = truth.sum().float()
        tot_true_positive = (truth.bool() & preds).sum().float()
        tot_auc = roc_auc_score(
            truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Eff, pur, auc
        sig_eff = sig_true_positive / sig_true
        sig_pur = sig_true_positive / edge_positive
        tot_eff = tot_true_positive / tot_true
        tot_pur = tot_true_positive / edge_positive

        # Combined metrics
        double_auc = sig_auc * tot_auc
        custom_f1 = 2 * sig_eff * tot_pur / (sig_eff + tot_pur)
        sig_fake_ratio = sig_true_positive / (edge_positive - tot_true_positive)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "val_loss": loss,
                    "current_lr": current_lr,
                    "sig_eff": sig_eff,
                    "sig_pur": sig_pur,
                    "sig_auc": sig_auc,
                    "tot_eff": tot_eff,
                    "tot_pur": tot_pur,
                    "tot_auc": tot_auc,
                    "double_auc": double_auc,
                    "custom_f1": custom_f1,
                    "sig_fake_ratio": sig_fake_ratio,
                },
                sync_dist=True,
            )

        return preds

    def shared_evaluation(self, batch, batch_idx, log=True):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        truth = (
            batch.y_pid.bool() if "y_pid" in self.hparams["regime"] else batch.y.bool()
        )
        edge_sample, truth_sample = self.handle_directed(batch, batch.edge_index, truth)

        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None

        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float().squeeze(), weight=manual_weights
        )

        preds = self.log_metrics(output, batch, loss, log)

        return {"loss": loss, "preds": preds, "score": torch.sigmoid(output)}

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)

        return outputs["loss"]

    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs

    def test_step_end(self, output_results):

        print("Step:", output_results)

    def test_epoch_end(self, outputs):

        print("Epoch:", outputs)

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
