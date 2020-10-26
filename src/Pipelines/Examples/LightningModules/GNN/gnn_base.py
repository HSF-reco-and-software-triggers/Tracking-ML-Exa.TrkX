import sys, os

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch

from .utils import load_dataset, random_edge_slice_v2


class GNNBase(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''
        # Assign hyperparameters
        self.hparams = hparams
        
        # Handle any subset of [train, val, test] data split, assuming that ordering
        input_dirs = [None, None, None]
        input_dirs[:len(hparams["datatype_names"])] = [os.path.join(hparams["input_dir"], datatype) for datatype in hparams["datatype_names"]]
        self.trainset, self.valset, self.testset = [load_dataset(input_dir, hparams["datatype_split"][i]) for i, input_dir in enumerate(input_dirs)]

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
        optimizer = [torch.optim.AdamW(self.parameters(), lr=(self.hparams["lr"]), betas=(0.9, 0.999), eps=1e-08, amsgrad=True)]
        scheduler = [
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], factor=self.hparams["factor"], patience=self.hparams["patience"]),
                'monitor': 'checkpoint_on',
                'interval': 'epoch',
                'frequency': 1
            }
        ]
#         scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=1, gamma=0.3)]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        
        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))

        subset_edges_ind, subset_edges_extended, nested_ind = random_edge_slice(self.hparams["delta_phi"], batch)
        
        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), 
                       batch.edge_index[:, subset_edges_extended]).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.edge_index[:, subset_edges_extended]).squeeze())

        if ('pid' in self.hparams["regime"]):
            y_pid = (batch.pid[batch.edge_index[0, subset_edges_ind]] == batch.pid[batch.edge_index[1, subset_edges_ind]]).float()
            loss = F.binary_cross_entropy_with_logits(output[nested_ind], y_pid.float(), pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output[nested_ind], batch.y[subset_edges_ind], pos_weight = weight)
            
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):

        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))

        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.edge_index).squeeze())

        if ('pid' in self.hparams["regime"]):
            y_pid = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            val_loss = F.binary_cross_entropy_with_logits(output, y_pid.float(), pos_weight = weight)
        else:
            val_loss = F.binary_cross_entropy_with_logits(output, batch.y, pos_weight = weight)

        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log('val_loss', val_loss)

        #Edge filter performance
        preds = F.sigmoid(output) > 0.5 #Maybe send to CPU??
        edge_positive = preds.sum().float()

        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
            edge_true = y_pid.sum().float()
            edge_true_positive = (y_pid & preds).sum().float()
        else:
            edge_true = batch.y.sum()
            edge_true_positive = (batch.y.bool() & preds).sum().float()

        result.log_dict({'eff': torch.tensor(edge_true_positive/edge_true), 'pur': torch.tensor(edge_true_positive/edge_positive)})

        return result

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step()
        optimizer.zero_grad()
