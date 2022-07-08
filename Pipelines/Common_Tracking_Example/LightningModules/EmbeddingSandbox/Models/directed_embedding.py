import torch.nn.functional as F

# 3rd party imports
from ..sandbox_base import SandboxDirectedEmbeddingBase

# Local imports
from ...GNN.utils import make_mlp

class DirectedEmbedding(SandboxDirectedEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        # Construct the MLP architecture
        self.network1 = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.network2 = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x1_out = self.network1(x)
        x2_out = self.network2(x)

        if "norm" in self.hparams["regime"]:
            return F.normalize(x1_out), F.normalize(x2_out)
        else:
            return x1_out, x2_out
