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

from .utils import load_dataset, purity_sample, LargeDataset
from sklearn.metrics import roc_auc_score


class HeteroGNNBase(LightningModule):
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
                self.trainset, batch_size=1, num_workers=4
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

    def handle_directed(self, batch, edge_sample, truth_sample, sample_indices):

        edge_sample = torch.cat([edge_sample, edge_sample.flip(0)], dim=-1)
        truth_sample = truth_sample.repeat(2)
        sample_indices = sample_indices.repeat(2)

        if ("directed" in self.hparams.keys()) and self.hparams["directed"]:
            direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
            edge_sample = edge_sample[:, direction_mask]
            truth_sample = truth_sample[direction_mask]

        return edge_sample, truth_sample, sample_indices
    
    def training_step(self, batch, batch_idx):

        truth = batch[self.hparams["truth_key"]]

        if ("train_purity" in self.hparams.keys()) and (
            self.hparams["train_purity"] > 0
        ):
            edge_sample, truth_sample, sample_indices = purity_sample(
                truth, batch.edge_index, self.hparams["train_purity"]
            )
        else:
            edge_sample, truth_sample, sample_indices = batch.edge_index, truth, torch.arange(batch.edge_index.shape[1])
            
        edge_sample, truth_sample, sample_indices = self.handle_directed(batch, edge_sample, truth_sample, sample_indices)

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth_sample.bool()).sum() / truth_sample.sum())
        )

        output = self(batch.x.float(), edge_sample, batch.volume_id).squeeze()

        if self.hparams["mask_background"]:
            y_subset = truth_sample | ~batch.y_pid[sample_indices].bool()
            output, truth_sample = output[y_subset], truth_sample[y_subset]

        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), pos_weight=weight
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def log_metrics(self, output, sample_indices, batch, loss, log):

        preds = torch.sigmoid(output) > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        sig_truth = batch.pid_signal[sample_indices]
        sig_true = sig_truth.sum().float()
        sig_true_positive = (sig_truth.bool() & preds).sum().float()
        sig_auc = roc_auc_score(
            sig_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Total true & total tp
        tot_truth = (batch.y_pid.bool() | batch.y.bool())[sample_indices]
        tot_true = tot_truth.sum().float()
        tot_true_positive = (tot_truth.bool() & preds).sum().float()
        tot_auc = roc_auc_score(
            tot_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
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

        truth = batch[self.hparams["truth_key"]]
        
        if ("train_purity" in self.hparams.keys()) and (
            self.hparams["train_purity"] > 0
        ):
            edge_sample, truth_sample, sample_indices = purity_sample(
                truth, batch.edge_index, self.hparams["train_purity"]
            )
        else:
            edge_sample, truth_sample, sample_indices = batch.edge_index, truth, torch.arange(batch.edge_index.shape[1])
            
        edge_sample, truth_sample, sample_indices = self.handle_directed(batch, edge_sample, truth_sample, sample_indices)

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth_sample.bool()).sum() / truth_sample.sum())
        )
        
        output = self(batch.x.float(), edge_sample, batch.volume_id).squeeze()

        if self.hparams["mask_background"]:
            y_subset = truth_sample | ~batch.y_pid[sample_indices].bool()
            subset_output, subset_truth_sample = output[y_subset], truth_sample[y_subset]
            loss = F.binary_cross_entropy_with_logits(
                subset_output, subset_truth_sample.float().squeeze(), pos_weight=weight
            )            
        else:
            loss = F.binary_cross_entropy_with_logits(
                output, truth_sample.float().squeeze(), pos_weight=weight
            )

        preds = self.log_metrics(output, sample_indices, batch, loss, log)

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
            self.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        
class LargeGNNBase(HeteroGNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering

        self.trainset, self.valset, self.testset = [
            LargeDataset(
                self.hparams["input_dir"],
                subdir,
                split,
                self.hparams
            )
            for subdir, split in zip(self.hparams["datatype_names"], self.hparams["datatype_split"])
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