import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.nn import Linear
import sys


class GNN_Base(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''
        # Assign hyperparameters
        self.hparams = hparams

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

    def shared_step(self, batch):
        weight = (torch.tensor((~batch.y.bool()).sum() / batch.y.sum()) if (self.hparams["weight"]==None)
                      else torch.tensor(self.hparams["weight"]))

        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.edge_index).squeeze())

        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
            loss = F.binary_cross_entropy_with_logits(output, y_pid.float(), pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y, pos_weight = weight)

        return output, loss

    def training_step(self, batch, batch_idx):

        _, loss = self.shared_step(batch)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):

        output, val_loss = self.shared_step(batch)

        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log('val_loss', val_loss)

        #Edge filter performance
        preds = F.sigmoid(output) > 0.5 #Maybe send to CPU??
        edge_positive = preds.sum().float()

        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
            edge_true = y_pid.sum()
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
