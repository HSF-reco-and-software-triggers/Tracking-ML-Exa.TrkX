import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch

from .utils import load_dataset


class GNNContract(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''
        # Assign hyperparameters
        self.hparams.update(hparams)
        self.hparams["posted_alert"] = False
        
    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering
        input_dirs = [None, None, None]
        input_dirs[:len(self.hparams["datatype_names"])] = [os.path.join(self.hparams["input_dir"], datatype) for datatype in self.hparams["datatype_names"]]
        self.trainset, self.valset, self.testset = [load_dataset(input_dir, self.hparams["datatype_split"][i], self.hparams["pt_min"]) for i, input_dir in enumerate(input_dirs)]
        
    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=8)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=8)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=8)
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
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=self.hparams["patience"], gamma=self.hparams["factor"]),
                'interval': 'epoch',
                'frequency': 1
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))
       
        edge_scores, cluster = self(batch.x, batch.edge_index)
        
        if 'weighting' in self.hparams['regime']:
            manual_weights = batch.weights
        else:
            manual_weights = None
        
        if ('pid' in self.hparams["regime"]):
            
            truth = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            
            ratio = torch.unique(cluster).size(0) / cluster.size(0)
            e_min = edge_scores.min(0, keepdim=True)[0].item()
            e_max = edge_scores.max(0, keepdim=True)[0].item()
            if e_min < 0:
                print(e_min)
            if e_max > 1:
                print(e_max)
            
            loss = F.binary_cross_entropy_with_logits(edge_scores, truth.float(), weight = manual_weights, pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y, weight = manual_weights, pos_weight = weight)
            
        self.log_dict({'train_loss': loss,'ratio':ratio})

        return loss

    def shared_evaluation(self, batch, batch_idx):
        
        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))

        edge_scores, cluster = self(batch.x, batch.edge_index)
        
        ratio = torch.unique(cluster).size(0) / cluster.size(0)
        
        clustered_edges_mask = (cluster[batch.edge_index[0]] == cluster[batch.edge_index[1]])
        clustered_edges = batch.edge_index[:,clustered_edges_mask]
        clustered_edge_score = edge_scores[clustered_edges_mask]
        clustered_edges_truth = (batch.pid[clustered_edges[0]] == batch.pid[clustered_edges[1]]).float()
        correct_edges_contracted = torch.sum(clustered_edges_truth).item() #total number of edges contracted that were supposed to be contracted
        edges_contracted = torch.sum(clustered_edges_mask).item() #total number of edges contracted
        
        
        truth = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
        e_min = edge_scores.min(0, keepdim=True)[0].item()
        e_max = edge_scores.max(0, keepdim=True)[0].item()
        if e_min < 0:
            print(e_min)
        if e_max > 1:
            print(e_max)
        
        if 'weighting' in self.hparams['regime']:
            manual_weights = batch.weights
        else:
            manual_weights = None
            
        loss = F.binary_cross_entropy_with_logits(edge_scores, truth, weight = manual_weights, pos_weight = weight)
        current_lr = self.optimizers().param_groups[0]['lr']
        
        eff = correct_edges_contracted/edges_contracted
        
        self.log_dict({'val_loss': loss, 'eff': eff, "current_lr": current_lr,"ratio":ratio})
        
        return {"loss": loss,"eff":eff, "truth": truth.cpu().numpy(),"ratio":ratio}

    def validation_step(self, batch, batch_idx):
        
        outputs = self.shared_evaluation(batch, batch_idx)
            
        return outputs["loss"]
    
    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx)
        
        return outputs
    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]
        
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
