import sys, os
import logging
from typing import Union

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch
import numpy as np

from .utils import load_dataset, purity_sample, LargeDataset, process_hetero_data, get_homo_data, get_hetero_data
from .plot_utils import plot_score
from .hetero_dataset import LargeHeteroDataset
from sklearn.metrics import roc_auc_score
from functools import partial
import wandb
import matplotlib.pyplot as plt

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
                self.trainset, batch_size=1, num_workers=2
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=1
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=1
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

        try:
            preds = self.log_metrics(output, sample_indices, batch, loss, log)
            return {"loss": loss, "preds": preds, "score": torch.sigmoid(output)}
        except:
            return {"loss": loss, "score": torch.sigmoid(output)}

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
                pg["lr"] = max(lr_scale * self.hparams["lr"], self.hparams.get('min_lr', 0))

        # update params
        optimizer.step(closure=optimizer_closure)
        if batch_idx==0 and self.hparams.get('debug'):
            print('='*20 + 'node encoder param grads')
            # print(self.node_encoders.parameters())
            # print('='*20 + 'node encoder param grads')
            # print(self.edge_encoders.parameters())
            for param in self.node_encoders.parameters():
                if not param.get_device() == 0: continue
                # print(param.grad.sum() if param.grad is not None else param.grad)
                print(param[0])
            
            print('='*20 +'edge encoder param grads')
            # # for enc in :
            for param in self.edge_encoders.parameters():
                # print(param.grad.sum() if param.grad is not None else param.grad)
                if not param.get_device() == 0: continue
                print(param[0])
            # print('='*20 +"convolution params grads")
            # for param in self.convs.parameters():
            #     # for param in enc.parameters():
            #     print(param.grad.sum() if param.grad is not None else param.grad)
            # print('='*20 +"updater params grads")
            # for param in self.edge_updaters.parameters():
            #     # for param in enc.parameters():
            #     print(param.grad.sum() if param.grad is not None else param.grad)
            # print('='*20 +'edge classifiers param grads')
            # for param in self.edge_classifiers.parameters():
            #     # for param in enc.parameters():
            #     print(param.grad.sum() if param.grad is not None else param.grad)
            # for param in self.parameters():
            #     print(param.grad.sum() if param.grad is not None else param.grad)
            
        optimizer.zero_grad()

class PyGHeteroGNNBase(HeteroGNNBase):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

    def setup(self, stage='fit'):
        # Handle any subset of [train, val, test] data split, assuming that ordering
        if self.trainset is None:
            print("Setting up dataset")
            self.trainset, self.valset, self.testset = [
                LargeHeteroDataset(
                    root=self.hparams['input_dir'],
                    subdir=subdir,
                    num_events=self.hparams["datatype_split"][i],
                    process_function=None,
                    hparams=self.hparams
                )
                for i, subdir in enumerate(self.hparams["datatype_names"])
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
                self.trainset, batch_size=1, num_workers=int(os.cpu_count()/4)
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=int(os.cpu_count()/4)
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=int(os.cpu_count()/4)
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None
    
    def training_step(self, batch, batch_idx, log=True):

        output_dict = self(batch)

        truth_dict = batch.collect(self.hparams['truth_key'])
        y_pid_dict = batch.collect('y_pid')
        losses = {}
        for key, o in output_dict.items():
            truth = truth_dict[key]
            y_pid = y_pid_dict[key]
            if self.hparams["mask_background"]:
                y_subset = truth | ~ y_pid.bool()
                o, truth = o[y_subset], truth[y_subset]
            weight = (
                torch.tensor(self.hparams["weight"])
                if ("weight" in self.hparams)
                else ((~truth.bool()).sum() / truth.sum()).clone() #torch.tensor((~truth.bool()).sum() / truth.sum())
            )
            key = "train_loss_" + key[0].replace('volume_', "") + '__' + key[2].replace('volume_', "" ) 
            losses[key] = F.binary_cross_entropy_with_logits(
                o,
                truth.float(),
                reduction='none', 
                pos_weight=weight
            )    
        if self.hparams.get('edge_weight'):
            loss = sum( [  torch.mean(l)  for l in losses.values()] ) / len(losses)
        else:
            loss = torch.mean( torch.cat(list(losses.values())) )


        # if self.hparams.get('edge_weight'):
        #     truth_dict = batch.collect(self.hparams['truth_key'])
        #     y_pid_dict = batch.collect('y_pid')
        #     losses = []
        #     for key, o in output_dict.items():

        #         truth = truth_dict[key]
        #         y_pid = y_pid_dict[key]
        #         if self.hparams["mask_background"]:
        #             y_subset = truth | ~ y_pid.bool()
        #             o, truth = o[y_subset], truth[y_subset]
        #         weight = (
        #             torch.tensor(self.hparams["weight"])
        #             if ("weight" in self.hparams)
        #             else ((~truth.bool()).sum() / truth.sum()).clone() #torch.tensor((~truth.bool()).sum() / truth.sum())
        #         )
        #         losses.append(
        #             F.binary_cross_entropy_with_logits(
        #                 o,
        #                 truth.to(torch.float32),
        #                 reduction='mean', 
        #                 pos_weight=weight
        #             )        
        #         )
        #     loss = sum(losses) / len(losses)
        # else:
        #     for key, o in output_dict.items():
        #         batch[key]['output'] = o
        #     batch = batch.to_homogeneous()

        #     truth = batch[self.hparams['truth_key']]
        #     output = batch.output
            
        #     weight = (
        #         torch.tensor(self.hparams["weight"])
        #         if ("weight" in self.hparams)
        #         else ((~truth.bool()).sum() / truth.sum()).clone() 
        #     )

        #     if self.hparams["mask_background"]:
        #         y_subset = truth | ~ batch.y_pid.bool()
        #         output, truth = output[y_subset], truth[y_subset]
            
        #     loss = F.binary_cross_entropy_with_logits(
        #         output,
        #         truth.to(torch.float32),
        #         reduction='mean', 
        #         pos_weight=weight
        #     )        
        if log:
            self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            if len(losses) > 1:
                self.log_dict(
                    {key: val.clone().detach().mean() for key, val in losses.items()},
                    on_epoch=True, on_step=False, batch_size=1, sync_dist=True
                )
            
        return {'loss': loss}

    def shared_evaluation(self, batch, batch_idx, log=True):

        output_dict = self(batch)
        for key, o in output_dict.items():
            batch[key]['output'] = o

        truth_dict = batch.collect(self.hparams['truth_key'])
        y_pid_dict = batch.collect('y_pid')
        losses = {}
        for key, o in output_dict.items():
            truth = truth_dict[key]
            y_pid = y_pid_dict[key]
            if self.hparams["mask_background"]:
                y_subset = truth | ~ y_pid.bool()
                o, truth = o[y_subset], truth[y_subset]
            weight = (
                torch.tensor(self.hparams["weight"])
                if ("weight" in self.hparams)
                else ((~truth.bool()).sum() / truth.sum()).clone() #torch.tensor((~truth.bool()).sum() / truth.sum())
            )
            key = "val_loss_" + key[0].replace('volume_', "") + '__' + key[2].replace('volume_', "" ) 
            losses[key] = F.binary_cross_entropy_with_logits(
                o,
                truth.to(torch.float32),
                reduction='none', 
                pos_weight=weight
            )
        if self.hparams.get('edge_weight'):
            loss = sum( [  torch.mean(l)  for l in losses.values()] ) / len(losses)
        else:
            loss = torch.mean( torch.cat(list(losses.values())) )    
                
        batch = batch.to_homogeneous()
        truth = batch[self.hparams['truth_key']]
        output = batch.output
        # else:
        #     batch = batch.to_homogeneous()
        #     truth = batch[self.hparams["truth_key"]]
        #     output = batch.output

        #     weight = (
        #         torch.tensor(self.hparams["weight"])
        #         if ("weight" in self.hparams)
        #         else ((~truth.bool()).sum() / truth.sum()).clone()
        #     )        

        #     if self.hparams["mask_background"]:
        #         y_subset = truth | ~ batch.y_pid.bool()
        #         loss = F.binary_cross_entropy_with_logits(
        #             output[y_subset], 
        #             truth[y_subset].to(torch.float32),
        #             reduction='mean', 
        #             pos_weight=weight
        #         ) 
        #     else:       
        #         loss = F.binary_cross_entropy_with_logits(
        #             output, 
        #             truth.to(torch.float32),
        #             reduction='mean', 
        #             pos_weight=weight
        #         )        
        if log:
            preds = self.log_metrics(output, truth, batch, loss, log, {}, {})
            if len(losses) > 1:
                self.log_dict(
                    {key: val.mean() for key, val in losses.items()},
                    on_epoch=True, on_step=False, batch_size=1, sync_dist=True
                )
        else:
            preds = torch.sigmoid(output) > self.hparams['edge_cut']
        return {"loss": loss, "preds": preds, "score": torch.sigmoid(output).clone().detach().cpu(), "truth": truth.clone().detach().cpu().float()}
        
    def log_metrics(self, output, truth, homo_batch, loss, log, output_dict, truth_dict):

        batch= homo_batch

        edge_score = torch.sigmoid(output)

        preds = edge_score > self.hparams["edge_cut"]

        # Positives
        edge_positive = preds.sum().float()

        # Signal true & signal tp
        sig_truth = batch[self.hparams['truth_key']]
        sig_true = sig_truth.sum().float()
        sig_true_positive = (sig_truth.bool() & preds).sum().float()
        sig_auc = roc_auc_score(
            sig_truth.bool().cpu().detach(), 
            edge_score.cpu().detach()
        )

        # Total true & total tp

        tot_truth = (batch.y_pid.bool() | batch.y.bool()) 
        tot_true = tot_truth.sum().float()
        tot_true_positive = (tot_truth.bool() & preds).sum().float()
        tot_auc = roc_auc_score(
            tot_truth.bool().cpu().detach(), preds.cpu().detach()
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
            batch_size=1,
            on_step=False,
            on_epoch=True
        )

        return preds
    
    def validation_step(self, batch, batch_idx):
        outputs = self.shared_evaluation(batch, batch_idx)
        return outputs
    
    def test_step(self, batch, batch_idx):
        outputs = self.shared_evaluation(batch, batch_idx, log=False)
        return outputs

    def log_plots(self, outputs, key, stage='val', *args, **kwargs):
        # plot score histogram for both classes
        if wandb.run is not None and self.current_epoch is not None and self.current_epoch % self.hparams.get('plot_every', 1) == 0:
            if 'score' in outputs[-1] and 'truth' in outputs[-1]:
                plot_score(outputs, stage, **kwargs)
                wandb.run.log({key: wandb.Image(plt)})
                plt.close()
    
    def validation_epoch_end(self, outputs) -> None:
        self.log_plots(outputs, stage='val', key='val_score_hist', title=f'Validation score histogram {self.current_epoch}', **self.hparams)
    
    def test_epoch_end(self, outputs, **kwargs):
        if 'score' in outputs[-1] and 'truth' in outputs[-1]:
            fig, ax = plot_score(outputs, stage='test', title="Test score histogram", **self.hparams)
            if kwargs.get('save_score_hist'):
                plt.savefig(kwargs['save_score_hist'])
            return fig, ax
        
    def on_train_start(self):
        self.trainer.strategy.optimizers = [self.trainer.lr_scheduler_configs[0].scheduler.optimizer]
        
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

