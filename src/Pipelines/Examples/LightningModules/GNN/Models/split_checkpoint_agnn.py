import torch

from .checkpoint_agnn import CheckpointedResAGNN
    
class SplitCheckpointedResAGNN(CheckpointedResAGNN):

    def __init__(self, hparams):
        super().__init__(hparams)
        
    def training_step(self, batch, batch_idx):
        
        weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams)
                      else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum()))

        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), 
                       batch.edge_index).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.edge_index).squeeze())
        
        if ('pid' in self.hparams["regime"]):
            y_pid = (batch.pid[batch.edge_index[0, batch.nested_ind[0]]] == batch.pid[batch.edge_index[1, batch.nested_ind[0]]]).float()
            loss = F.binary_cross_entropy_with_logits(output[batch.nested_ind[0]], y_pid.float(), pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output[batch.nested_ind[0]], batch.y[batch.nested_ind[0]], pos_weight = weight)
                        
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)

        return result
        