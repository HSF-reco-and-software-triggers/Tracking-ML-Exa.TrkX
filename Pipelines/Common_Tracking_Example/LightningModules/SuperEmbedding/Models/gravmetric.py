# 3rd party imports
from ..multiembedding_base import MultiEmbeddingBase
from ..gravmetric_base import GravMetricBase
from torch_scatter import scatter_mean, scatter_add
import numpy as np
import torch
import torch.nn.functional as F

# Local imports
from ...Embedding.utils import build_edges, build_knn
from ...GNN.utils import make_mlp

class UndirectedGravMetric(MultiEmbeddingBase):
    def __init__(self, hparams):
        print("UndirectedGravMetric")
        super().__init__(hparams)
        """
        An implementation of the GravMetric architecture: The most naive version,
        such that input maps are undirected, and there is only topo-space.

        Behaviour is:
        1. topo = map_0(x)
        2. Get neighbourhoods as edge list
        3. Weight start_node topo by edge potential
        4. Scatter_mean weighted start_node topo at end_nodes
        5. Pass [weighted_topo, end_nodes] to map_1(x)
        """

        # Construct the MLP architecture
        self.map_0 = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        if hparams["feature_hidden"] > 0:
            self.feature_mlp = make_mlp(
                hparams["spatial_channels"] + hparams["cell_channels"],
                [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["feature_hidden"]],
                hidden_activation=hparams["activation"],
                output_activation=None,
                layer_norm=True,
            )
            feature_size = hparams["feature_hidden"]
        else:
            feature_size = hparams["emb_dim"]

        self.map_1 = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"] + feature_size,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        # 1. topo = map_0(x)
        topo = F.normalize(self.map_0(x))

        # 2. Get neighbourhoods as edge list
        # topo_edges = build_edges(topo, topo, r_max=self.hparams["topo_margin"], k_max=self.hparams["topo_k"])
        topo_edges = build_knn(topo, self.hparams["topo_k"])

        # 3. Weight start_node topo by edge potential
        edge_potentials = self.get_potential(topo, topo_edges)
        if self.hparams["feature_hidden"] > 0:
            features = self.feature_mlp(x)
            weighted_features = features[topo_edges[0]] * edge_potentials.unsqueeze(-1)
        else:
            weighted_features = topo[topo_edges[0]] * edge_potentials.unsqueeze(-1)
            
        # 4. Scatter_mean weighted start_node topo at end_nodes
        mean_neighborhood = scatter_mean(weighted_features, topo_edges[1], dim=0, dim_size=topo.shape[0])

        # 5. Pass [weighted_topo, end_nodes] to map_1(x)
        return topo, F.normalize(self.map_1(torch.cat([x, mean_neighborhood], dim=-1)))

    def get_potential(self, x, edges):

        d_sq = ((x[edges[0]] - x[edges[1]])**2).sum(dim=-1)

        potential = (torch.exp(1 - d_sq / self.hparams["topo_margin"]**2) - 1) / (np.exp(1) - 1)

        return potential        

class DirectedGravMetric(GravMetricBase):
    def __init__(self, hparams):
        print("DirectedGravMetric")
        super().__init__(hparams)
        """
        An implementation of the GravMetric architecture: The most naive version,
        such that input maps are undirected, and there is only topo-space.

        Behaviour is:
        1. topo_a, topo_b  = map_0a(x), map_0b(x)
        2. Get neighbourhoods as edge list (from a->b)
        3. Weight all nodes topo_a and topo_b by edge potential
        4. Scatter_mean weighted start_node topo_a at end_nodes, and weight end_node topo_b at start_nodes
        5. Pass [mean_neighbors_a, topo_a, topo_b, mean_neighbors_b] to map_1a, map_1b
        """

        if "node_update" not in hparams:
            self.hparams["node_update"] = "feat"
            hparams["node_update"] = "feat"

        # Construct the MLP architecture
        self.map_0a, self.map_0b = [make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        ) for _ in range(2)]

        if hparams["feature_hidden"] > 0 and (self.hparams["node_update"] in ["feat", "topofeat"]):
            self.feature_mlp_a, self.feature_mlp_b = [make_mlp(
                hparams["spatial_channels"] + hparams["cell_channels"],
                [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["feature_hidden"]],
                hidden_activation=hparams["activation"],
                output_activation=None,
                layer_norm=True,
            ) for _ in range(2)]
        
        if self.hparams["node_update"] == "feat":
            feature_size = 2 * hparams["feature_hidden"] + hparams["spatial_channels"] + hparams["cell_channels"]
        elif self.hparams["node_update"] == "topo":
            feature_size = 2 * hparams["emb_dim"] + hparams["spatial_channels"] + hparams["cell_channels"]
        elif self.hparams["node_update"] == "topofeat":
            feature_size = 2 * hparams["emb_dim"] + 2 * hparams["feature_hidden"] + hparams["spatial_channels"] + hparams["cell_channels"]

        self.map_1a, self.map_1b = [make_mlp(
            feature_size,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        ) for _ in range(2)]

        self.save_hyperparameters()

    def forward(self, x):

        # 1. Embed and build topo graph
        topo_a, topo_b = F.normalize(self.map_0a(x)), F.normalize(self.map_0b(x))
        topo_edges = build_edges(topo_a, topo_b, r_max=self.hparams["topo_margin"], k_max=self.hparams["topo_k"])

        # 2. Get gravity potentials and weight features as attention mechanism
        edge_potentials = self.get_potential(topo_a, topo_b, topo_edges)
        if self.hparams["feature_hidden"] > 0 and (self.hparams["node_update"] in ["feat", "topofeat"]):
            features_in = self.feature_mlp_a(x)
            features_out = self.feature_mlp_b(x)
            if self.hparams["node_update"] == "feat":
                weighted_features_in = features_in[topo_edges[0]] * edge_potentials.unsqueeze(-1)
                weighted_features_out = features_out[topo_edges[1]] * edge_potentials.unsqueeze(-1)
            elif self.hparams["node_update"] == "topofeat":
                weighted_features_in = torch.cat([features_in, topo_a], dim=-1)[topo_edges[0]] * edge_potentials.unsqueeze(-1)
                weighted_features_out = torch.cat([features_out, topo_b], dim=-1)[topo_edges[1]]  * edge_potentials.unsqueeze(-1)
        else:
            weighted_features_in = topo_a[topo_edges[0]] * edge_potentials.unsqueeze(-1)
            weighted_features_out = topo_b[topo_edges[1]] * edge_potentials.unsqueeze(-1)
            
        if self.hparams["aggregation"] == "mean":
            neighborhood_in = scatter_mean(weighted_features_in, topo_edges[1], dim=0, dim_size=topo_a.shape[0]) 
            neighborhood_out = scatter_mean(weighted_features_out, topo_edges[0], dim=0, dim_size=topo_b.shape[0]) 
        else:
            neighborhood_in = scatter_add(weighted_features_in, topo_edges[1], dim=0, dim_size=topo_a.shape[0])  
            neighborhood_out = scatter_add(weighted_features_out, topo_edges[0], dim=0, dim_size=topo_b.shape[0])
            
            if self.hparams["aggregation"] == "weighted_sum":
                mean_weights_in = scatter_add(edge_potentials, topo_edges[1], dim=0, dim_size=topo_a.shape[0])
                neighborhood_in[mean_weights_in != 0] = neighborhood_in[mean_weights_in != 0] / mean_weights_in[mean_weights_in != 0].unsqueeze(-1)
                mean_weights_out = scatter_add(edge_potentials, topo_edges[0], dim=0, dim_size=topo_b.shape[0])
                neighborhood_out[mean_weights_out != 0] = neighborhood_out[mean_weights_out != 0] / mean_weights_out[mean_weights_out != 0].unsqueeze(-1)

        # 4. Pass [mean_neighbors_a, topo_a, topo_b, mean_neighbors_b] to map_1a, map_1b
        out_a = F.normalize(self.map_1a(torch.cat([neighborhood_in, x, neighborhood_out], dim=-1)))
        out_b = F.normalize(self.map_1b(torch.cat([neighborhood_in, x, neighborhood_out], dim=-1)))

        return topo_a, topo_b, out_a, out_b

    def get_potential(self, x_a, x_b, edges):

        d_sq = ((x_a[edges[0]] - x_b[edges[1]])**2).sum(dim=-1)

        potential = (torch.exp(1 - d_sq / self.hparams["topo_margin"]**2) - 1) / (np.exp(1) - 1)

        return potential        