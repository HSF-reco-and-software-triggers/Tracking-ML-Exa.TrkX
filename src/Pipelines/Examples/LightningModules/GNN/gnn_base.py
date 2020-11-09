import sys, os

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
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
        self.hparams["posted_alert"] = False
        
        # Handle any subset of [train, val, test] data split, assuming that ordering
        input_dirs = [None, None, None]
        input_dirs[:len(hparams["datatype_names"])] = [os.path.join(hparams["input_dir"], datatype) for datatype in hparams["datatype_names"]]
        self.trainset, self.valset, self.testset = [load_dataset(input_dir, hparams["datatype_split"][i], hparams["pt_min"]) for i, input_dir in enumerate(input_dirs)]
        print("Data processed")
        
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
#         scheduler = [
#             {
#                 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], factor=self.hparams["factor"], patience=self.hparams["patience"]),
#                 'monitor': 'val_loss',
#                 'interval': 'epoch',
#                 'frequency': 1
#             }
#         ]
        scheduler = [
            {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=50, gamma=0.3),
                'interval': 'epoch',
                'frequency': 1
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        
        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))
        
        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), 
                       batch.edge_index).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.edge_index).squeeze())

        if ('pid' in self.hparams["regime"]):
            y_pid = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            loss = F.binary_cross_entropy_with_logits(output, y_pid.float(), pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y, pos_weight = weight)
            
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))

        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.edge_index).squeeze())

        if ('pid' in self.hparams["regime"]):
            y_pid = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            loss = F.binary_cross_entropy_with_logits(output, y_pid.float(), pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y, pos_weight = weight)

        #Edge filter performance
        preds = F.sigmoid(output) > self.hparams["edge_cut"]
        edge_positive = preds.sum().float()

        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
            edge_true = y_pid.sum().float()
            edge_true_positive = (y_pid & preds).sum().float()
        else:
            edge_true = batch.y.sum()
            edge_true_positive = (batch.y.bool() & preds).sum().float()

        eff = torch.tensor(edge_true_positive/edge_true)
        pur = torch.tensor(edge_true_positive/edge_positive)
        
        if (eff > 0.99) and (pur > 0.99) and not self.hparams["posted_alert"] and self.hparams["slack_alert"]:
            self.logger.experiment.alert(title="High Performance", 
                        text="Efficiency and purity have both cracked 99%. Great job, Dan! You're having a great Thursday, and I think you've earned a celebratory beer.",
                        wait_duration=timedelta(minutes=60))
            self.hparams["posted_alert"] = True
            
        current_lr = self.optimizers().param_groups[0]['lr']
        print(current_lr)
        self.log_dict({'val_loss': loss, 'eff': eff, 'pur': pur, "current_lr": current_lr})
            
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]
        
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
