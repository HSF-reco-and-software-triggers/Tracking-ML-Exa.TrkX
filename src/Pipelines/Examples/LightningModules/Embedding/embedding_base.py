# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_cluster import radius_graph
import numpy as np

# Local Imports
from .utils import load_dataset, graph_intersection
if torch.cuda.is_available():
    from .utils import build_edges, res
    device = 'cuda'
else:
    device = 'cpu'

def split_datasets(input_dir, train_split, pt_cut = 0, seed = 1):
    '''
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    '''
    torch.manual_seed(seed)
    loaded_events = load_dataset(input_dir, sum(train_split), pt_cut)
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events

class EmbeddingBase(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different embedding training regimes
        '''
        # Assign hyperparameters
        self.hparams = hparams
        
    def setup(self, stage):
        if stage == "fit":
            self.trainset, self.valset, self.testset = split_datasets(self.hparams["input_dir"], self.hparams["train_split"], self.hparams["pt_min"])

    def train_dataloader(self):
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [torch.optim.AdamW(self.parameters(), lr=(self.hparams["lr"]), betas=(0.9, 0.999), eps=1e-08, amsgrad=True)]
        scheduler = [
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], factor=self.hparams["factor"], patience=self.hparams["patience"]),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        ]
#         scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=1, gamma=0.3)]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        """
        Example:
        TODO - Explain how the embedding training step works by example!

        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        if 'ci' in self.hparams["regime"]:
            spatial = self(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = self(batch.x)

        e_bidir = torch.cat([batch.layerless_true_edges, batch.layerless_true_edges.flip(0)], axis=-1)

        e_spatial = torch.empty([2,0], dtype=torch.int64, device=self.device)

        if 'rp' in self.hparams["regime"]:
        # Get random edge list
            n_random = int(self.hparams["randomisation"]*e_bidir.shape[1])
#             e_spatial = torch.cat([e_spatial, torch.randint(int(e_bidir.min().item()), int(e_bidir.max().item()), (2, n_random), device=self.device)], axis=-1)
            e_spatial = torch.cat([e_spatial, torch.randint(e_bidir.min(), e_bidir.max(), (2, n_random), device=self.device)], axis=-1)

        if 'hnm' in self.hparams["regime"]:
            e_spatial = (torch.cat([e_spatial, build_edges(spatial, self.hparams["r_train"], self.hparams["knn"], res)], axis=-1)
                        if torch.cuda.is_available()
                        else torch.cat([e_spatial, radius_graph(spatial, r=self.hparams["r_train"], max_num_neighbors=self.hparams["knn"])], axis=-1))

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        e_spatial = torch.cat([e_spatial, e_bidir.transpose(0,1).repeat(1,self.hparams["weight"]).view(-1, 2).transpose(0,1)], axis=-1)
        y_cluster = np.concatenate([y_cluster.astype(int), np.ones(e_bidir.shape[1]*self.hparams["weight"])])

        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=self.hparams["margin"], reduction="mean")

        self.log('train_loss', loss)

        return loss
    
    
    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):
        
        if 'ci' in self.hparams["regime"]:
            spatial = self(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = self(batch.x)
            
        e_bidir = torch.cat([batch.layerless_true_edges, batch.layerless_true_edges.flip(0)], axis=-1)

        # Build whole KNN graph
        e_spatial = (build_edges(spatial, knn_radius, knn_num, res)
                    if torch.cuda.is_available()
                    else radius_graph(spatial, r = knn_radius, max_num_neighbors=1000))

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)
        
        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=self.hparams["margin"], reduction="mean")

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])
        
        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        if log:
            self.log_dict({'val_loss': loss, 'eff': eff, 'pur': pur, "current_lr": current_lr})
        print(eff, pur)
        print(batch.event_file)
        
        return {"loss": loss, "preds": e_spatial.cpu().numpy(), "truth": y_cluster, "truth_graph": e_bidir.cpu().numpy()}
        
    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """

        outputs = self.shared_evaluation(batch, batch_idx, self.hparams["r_val"], 1000, log=True)

        return outputs["loss"]
    
    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, self.hparams["r_test"], 1000, log=False)

        return outputs

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]
        
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
