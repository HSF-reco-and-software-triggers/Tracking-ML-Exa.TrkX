import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch
from torch_scatter import scatter_add,scatter_max, scatter_mean
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
       
        nodes, edge_scores, cluster = self(batch.x, batch.edge_index)
        
        if 'weighting' in self.hparams['regime']:
            manual_weights = batch.weights
        else:
            manual_weights = None
        
        if ('pid' in self.hparams["regime"]):
            #print("nodes",nodes._version)
            edge_truth = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            edge_cluster_mask = cluster[batch.edge_index[0]] == cluster[batch.edge_index[1]]
            node_truth = scatter_mean(batch.pt,cluster,dim_size = nodes.size(0))
            #print("cluster",cluster.shape)
            #print("node truth",node_truth.shape)
            #print("pt",batch.pt.shape)
            
            edges_clustered = cluster[batch.edge_index[:,edge_cluster_mask]]
            truth_clustered = edge_truth[edge_cluster_mask].float()
            #print("truth clustered",truth_clustered.shape)
            scores_clustered = edge_scores[edge_cluster_mask]
            #print("scores clustered",scores_clustered.shape)
            truth_sum = scatter_add(truth_clustered,edges_clustered[0])
            score_sum = scatter_add(scores_clustered,edges_clustered[0])
            ratio = torch.unique(cluster).size(0) / cluster.size(0)
            clustered_loss = 0.0
            edge_loss = 0.0
            node_loss = 0.0
            if "node" not in self.hparams["regime"]:
                clustered_loss = F.mse_loss(score_sum, truth_sum)
                edge_loss = F.mse_loss(edge_scores,edge_truth)
            if "hybrid" in self.hparams["regime"] or "node" in self.hparams["regime"]:
                nodes = nodes.squeeze()
                #print(cluster._version)
                #print(node_truth._version)
                node_loss = F.mse_loss(nodes.squeeze(),node_truth)
            loss = clustered_loss + node_loss + edge_loss
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y, weight = manual_weights, pos_weight = weight)
            
        self.log_dict({'train_loss': loss,'ratio':ratio})

        return loss

    def shared_evaluation(self, batch, batch_idx):
        
        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))

        nodes, edge_scores, cluster = self(batch.x, batch.edge_index)
        
        if 'weighting' in self.hparams['regime']:
            manual_weights = batch.weights
        else:
            manual_weights = None
        
        node_truth = scatter_mean(batch.pt,cluster)
        edge_truth = (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
        edge_cluster_mask = cluster[batch.edge_index[0]] == cluster[batch.edge_index[1]]
        edge_pred_clustered = cluster[batch.edge_index[:,edge_cluster_mask]]
        edge_truth_clustered = edge_truth[edge_cluster_mask].float()
        edge_scores_clustered = edge_scores[edge_cluster_mask]
        edge_truth_sum = scatter_add(edge_truth_clustered,edge_pred_clustered[0])
        edge_score_sum = scatter_add(edge_scores_clustered,edge_pred_clustered[0])
        
        ratio = torch.unique(cluster).size(0) / cluster.size(0)
        clustered_loss = 0.0
        edge_loss = 0.0
        node_loss = 0.0
        if "node" not in self.hparams["regime"]:
            clustered_loss = F.mse_loss(edge_score_sum, edge_truth_sum)
            edge_loss = F.mse_loss(edge_scores,edge_truth)
        if "hybrid" in self.hparams["regime"] or "node" in self.hparams["regime"]:
            #print(nodes._version)
            node_loss = F.mse_loss(nodes.squeeze(),node_truth)
        loss = clustered_loss + node_loss + edge_loss
        num_edges_contracted = torch.sum(edge_cluster_mask)
        true_pred_edges = torch.sum(edge_truth_clustered)
        true_total_edges = torch.sum(edge_truth)
    
        current_lr = self.optimizers().param_groups[0]['lr']
        
        pur = true_pred_edges / num_edges_contracted
        eff = true_pred_edges / true_total_edges
        self.log_dict({'val_loss': loss, 'eff': eff, "pur":pur,"current_lr": current_lr,"ratio":ratio})
        
        return {"loss": loss,"eff":eff, "truth": edge_truth.cpu().numpy(),"ratio":ratio}

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
