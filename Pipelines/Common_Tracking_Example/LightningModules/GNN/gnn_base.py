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

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering
        input_dirs = [None, None, None]
        input_dirs[: len(self.hparams["datatype_names"])] = [
            os.path.join(self.hparams["input_dir"], datatype)
            for datatype in self.hparams["datatype_names"]
        ]
        self.trainset, self.valset, self.testset = [
            load_dataset(
                input_dir, 
                self.hparams["datatype_split"][i], 
                self.hparams["pt_background_min"],
                self.hparams["pt_signal_min"],
                self.hparams["true_edges"],
                self.hparams["noise"]
            )
            for i, input_dir in enumerate(input_dirs)
        ]
        
        if (self.trainer) and ("logger" in self.trainer.__dict__.keys()) and ("_experiment" in self.logger.__dict__.keys()):
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")

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

    def get_input_data(self, batch):
        
        if self.hparams["cell_channels"] > 0:
            print(batch.cell_data.shape)
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], batch.x], axis=-1)
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

        if ("train_purity" in self.hparams.keys()) and (self.hparams["train_purity"] > 0):
            edge_sample, truth_sample = purity_sample(batch, self.hparams["train_purity"], self.hparams["regime"])
        else:
            edge_sample, truth_sample = batch.edge_index, batch.y_pid
        
        edge_sample, truth_sample = self.handle_directed(batch, edge_sample, truth_sample)
        
        # Handle training towards a subset of the data
        if "subset" in self.hparams["regime"]:
            subset_mask = np.isin(edge_sample.cpu(), batch.modulewise_true_edges.unique().cpu()).any(0)
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

        truth = truth_sample.float() if "pid" in self.hparams["regime"] else batch.y
        
        loss = F.binary_cross_entropy_with_logits(
            output, truth.float(), weight=manual_weights, pos_weight=weight
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)
                
        return loss

    def get_metrics(self, truth, output):
        
        predictions = torch.sigmoid(output) > self.hparams["edge_cut"]
        
        edge_positive = predictions.sum().float()
        edge_true = truth.sum().float()
        edge_true_positive = (truth.bool() & predictions).sum().float()        

        eff = edge_true_positive / edge_true
        pur = edge_true_positive / edge_positive

        auc = roc_auc_score(truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach())
        
        return predictions, eff, pur, auc
    
    def shared_evaluation(self, batch, batch_idx, log=True):

        if "subset" in self.hparams["regime"]:
            subset_mask = np.isin(batch.edge_index.cpu(), batch.modulewise_true_edges.unique().cpu()).any(0)
        else:
            subset_mask = torch.ones(batch.edge_index.shape[1]).bool()
            
        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        input_data = self.get_input_data(batch)
        
        edge_sample, truth_sample = self.handle_directed(batch, batch.edge_index, batch.y_pid)
        output = self(input_data, edge_sample).squeeze()

        truth = truth_sample if "pid" in self.hparams["regime"] else batch.y.squeeze()

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None

        loss = F.binary_cross_entropy_with_logits(
            output, truth.float().squeeze(), weight=manual_weights
        )

        predictions, eff, pur, auc = self.get_metrics(truth, output)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "auc": auc, "current_lr": current_lr}
            )

        return {
            "loss": loss,
            "preds": predictions,
            "score": torch.sigmoid(output),
            "truth": truth,
        }

    
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
