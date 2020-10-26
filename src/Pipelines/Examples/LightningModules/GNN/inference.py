import sys, os

from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import torch

'''
Class-based Callback inference for integration with Lightning
'''

class GNNInferenceCallback(Callback):
    def __init__(self):
        self.output_dir = None
        self.overwrite = False

    def on_test_start(self, trainer, pl_module):
        # Prep the directory to produce inference data to
        self.output_dir = pl_module.hparams.output_dir
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True) for datatype in self.datatypes]

    def on_test_end(self, trainer, pl_module):
        print("Training finished, running inference to filter graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {"train": pl_module.trainset, "val": pl_module.valset, "test": pl_module.testset}
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0

        pl_module.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for batch_idx, batch in enumerate(dataset):
                    percent = (batch_incr / total_length) * 100
                    sys.stdout.flush()
                    sys.stdout.write(f'{percent:.01f}% inference complete \r')
                    if (not os.path.exists(os.path.join(self.output_dir, datatype, batch.event_file[-4:]))) or self.overwrite:
                        batch = batch.to(pl_module.device) #Is this step necessary??
                        batch = self.construct_downstream(batch, pl_module)
                        self.save_downstream(batch, pl_module, datatype)

                    batch_incr += 1

    def construct_downstream(self, batch, pl_module):

        emb = (None if (pl_module.hparams["emb_channels"] == 0)
               else batch.embedding)  # Does this work??

        output = pl_module(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius, emb).squeeze() if ('ci' in pl_module.hparams["regime"]) else pl_module(batch.x, batch.e_radius, emb).squeeze()
        y_pid = batch.pid[batch.e_radius[0]] == batch.pid[batch.e_radius[1]]

        cut_indices = F.sigmoid(output) > pl_module.hparams["filter_cut"]
        batch.e_radius = batch.e_radius[:, cut_indices]
        batch.y_pid = y_pid[cut_indices]
        batch.y = batch.y[cut_indices]

        return batch

    def save_downstream(self, batch, pl_module, datatype):

        with open(os.path.join(self.output_dir, datatype, batch.event_file[-4:]), 'wb') as pickle_file:
            torch.save(batch, pickle_file)
            
