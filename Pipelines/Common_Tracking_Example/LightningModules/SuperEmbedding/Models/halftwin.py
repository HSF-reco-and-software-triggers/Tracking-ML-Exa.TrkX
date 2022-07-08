# 3rd party imports
from ..super_embedding_base import SuperEmbeddingBase
import torch
import torch.nn.functional as F

# Local imports
from ...GNN.utils import make_mlp
from ...Embedding.utils import build_edges


class UndirectedHalfTwinEmbedding(SuperEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

        # Construct the MLP architecture
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]

        torch.manual_seed(0)

        self.net1 = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation="Tanh",
            output_activation=None,
            layer_norm=True,
        )

        self.net2 = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation="Tanh",
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        x1_out = self.net1(x)
        x2_out = self.net2(x)
        
        if "norm" in self.hparams["regime"]:
            return F.normalize(x1_out), F.normalize(x2_out)
        else:
            return x1_out, x2_out

class DirectedHalfTwinEmbedding(UndirectedHalfTwinEmbedding):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        spatial1, spatial2 = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial1)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial2)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial2)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = batch.signal_true_edges

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, e_bidir
        )

        hinge, d = self.get_hinge_distance(spatial1, spatial2, e_spatial, y_cluster)

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"]**2,
            reduction="mean",
        )

        loss = negative_loss + self.hparams["weight"] * positive_loss

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        input_data = self.get_input_data(batch)
        spatial1, spatial2 = self(input_data)

        e_bidir = batch.signal_true_edges

        # Build whole KNN graph
        e_spatial = build_edges(
            spatial1, spatial2, indices=None, r_max=knn_radius, k_max=knn_num
        )

        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        hinge, d = self.get_hinge_distance(
            spatial1, spatial2, e_spatial.to(self.device), y_cluster
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr}
            )

        return {
            "loss": loss,
            #             "distances": d,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
        }