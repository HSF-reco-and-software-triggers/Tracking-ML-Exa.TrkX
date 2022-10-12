import torch.nn.functional as F
import torch

# 3rd party imports
from ..sandbox_base import SandboxEmbeddingBase

# Local imports
from ...GNN.utils import make_mlp

class UndirectedEmbedding(SandboxEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        self.network = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, batch):
        
        x = torch.cat([batch.x, batch.cell_data[:, : self.hparams["cell_channels"]]], axis=-1)
        x_out = self.network(x)
        x_out = F.normalize(x_out) if "norm" in self.hparams["regime"] else x_out

        return x_out
