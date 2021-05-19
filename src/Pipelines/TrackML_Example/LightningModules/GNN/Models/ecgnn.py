import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add,scatter_max
from torch.utils.checkpoint import checkpoint
from torch_sparse import coalesce

from ..gnn_contract import GNNContract
from ..utils import make_mlp
from ..EdgePooling import EdgePooling


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, nb_layers, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2, [output_dim]*nb_layers,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
#         mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
#         mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
#         node_inputs = torch.cat([mi, mo, x], dim=1)
        messages = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0]) + scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([messages, x], dim=1)
        return self.network(node_inputs)
       
class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim, nb_layers, hidden_activation='Tanh',
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim]*nb_layers+[1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)
    
    def __merge_edges__(self, x, edge_index, batch, edge_score):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nodes = torch.arange(x.shape[0])
        nodes = nodes.to(device)

        nodes_remaining = torch.ones_like(nodes,dtype = torch.bool)
        nodes_remaining = nodes_remaining.to(device)
        edges_remaining = torch.ones_like(edge_index[0],dtype=torch.bool)
        edges_remaining = edges_remaining.to(device)
        ratio = 1.0
        i = 0
        old_edge_score = edge_score[:]
        while i < 10 and ratio > 0.05:    
            #get max edge score for each node and edge index where it occurs
            max_score_0, max_indices_0 = scatter_max(edge_score, edge_index[0], dim=0, dim_size=x.shape[0])
            max_score_1, max_indices_1 = scatter_max(edge_score, edge_index[1], dim=0, dim_size=x.shape[0])

            #stack scores for each direction
            stacked_score, stacked_indices = torch.stack([max_score_0, max_score_1]), torch.stack([max_indices_0, max_indices_1]).T
            top_score , _ = torch.max(stacked_score, dim=0)
            top_score = top_score.to(device)
            print(edge_index.shape)
            print(torch.min(edge_index),torch.max(edge_index))
            #get max neighbor for each node
            max_indices = torch.zeros(len(top_score), dtype=torch.long, device=device)
            max_indices[max_score_0 > max_score_1] = edge_index[1][max_indices_0[max_score_0 > max_score_1]]
            max_indices[max_score_1 > max_score_0] = edge_index[0][max_indices_1[max_score_1 > max_score_0]]

            #find edges where each node is the other's max index
            edge_index_match_0 = max_indices[edge_index[1]] == edge_index[0]
            edge_index_match_1 = max_indices[edge_index[0]] == edge_index[1]
            node_0_valid = nodes_remaining[edge_index[0]]
            node_1_valid = nodes_remaining[edge_index[1]]
            edge_index_match = edge_index_match_0 & edge_index_match_1 & node_0_valid & node_1_valid

            #update the remaining edges based on which ones should be removed
            edges_remaining &= ~edge_index_match
            edges_contracted = edge_index[:,edge_index_match]

            #update the remaining nodes based on which ones should be removed
            nodes_removed = torch.flatten(edges_contracted)
            nodes_remaining[nodes_removed] = 0.0

            #zero out the edge scores of every edge that has >= 1 node being removed
            edge_score_zero_mask = (edge_index[..., None] == nodes[nodes_removed]).any(-1).any(0)
            edge_score *= ~edge_score_zero_mask
            ratio = (torch.sum(nodes_remaining)/nodes_remaining.shape[0]).item()
            i += 1
            
        #split into edges removed and edges not removed
        edges_contracted = edge_index[:,~edges_remaining]
        new_e = edge_index[:,edges_remaining]
        
        #sort nodes into new ordering by cluster
        clustered_indices = torch.arange(edges_contracted.shape[1]).to(device)
        remaining_indices = edges_contracted.shape[1] + torch.arange(torch.sum(nodes_remaining)).to(device)
        new_node_index_map = torch.cat([
            torch.stack([edges_contracted[0],clustered_indices]),
            torch.stack([edges_contracted[1],clustered_indices]),
            torch.stack([nodes[nodes_remaining],remaining_indices])],dim=-1)
        new_node_index_map = new_node_index_map[:,torch.argsort(new_node_index_map[0])]
        cluster = new_node_index_map[1,:]
        
        
        #count the number of occurences of each node to find duplicates
        _, counts = torch.unique(new_node_index_map[0], return_counts=True)
        duplicates = torch.where(counts >= 2)
        cluster_mask = torch.ones_like(cluster,dtype=torch.bool)
        
        #if it finds a duplicate node, remove it from the cluster map
        if duplicates[0].size(0) > 0:
            d = duplicates[0]
            for i in d:
                #find the multiple cluster indices in the node index map and get the minimum index to keep
                duplicate_mask = (i == new_node_index_map[0])
                duplicate_cluster_indices = cluster[duplicate_mask]
                valid_cluster_index = torch.min(duplicate_cluster_indices)
                
                #mask all cluster indices that need removing
                cluster_remove_mask = ~(duplicate_mask & (cluster != valid_cluster_index))
                cluster_mask &= cluster_remove_mask
            
            #get all unique clusters that are being removed, and decrease all clusters greater than that index by 1 to account for each cluster removal
            clusters_to_remove = torch.unique(cluster[~cluster_mask])
            cluster = cluster[cluster_mask]
            for c in clusters_to_remove:
                cluster[cluster > c] -= 1
                
        #create new node features and edge index based on clustering
        new_x = scatter_add(x, cluster, dim=0, dim_size=torch.unique(cluster).shape[0])
        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[new_e], None, N, N)
        contracted_scores = old_edge_score[~edges_remaining]
        remaining_scores = torch.ones(N - edges_contracted.size(1),device=device)
        new_edge_score = torch.cat([contracted_scores,remaining_scores])
        new_x = new_x * new_edge_score.view(-1,1)
    
        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        batch = batch.to(x.device)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)
        
        return new_x, new_edge_index, new_batch, old_edge_score, unpool_info


class ECGNN(GNNContract):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different GNN training regimes
        '''

        # Setup input network
        self.input_network = make_mlp(hparams["in_channels"], [hparams["hidden"]],
                                      output_activation=hparams["hidden_activation"],
                                      layer_norm=hparams["layernorm"])
        # Setup the node layers
        self.node_network = NodeNetwork(hparams["in_channels"] + hparams["hidden"], hparams["hidden"],
                                        hparams["nb_node_layer"], hparams["hidden_activation"], hparams["layernorm"])
        
        # Setup the edge pooling layers
        self.edge_network = EdgeNetwork(hparams["in_channels"] + hparams["hidden"], hparams["in_channels"] + hparams["hidden"],
                                        hparams["nb_edge_layer"], hparams["hidden_activation"], hparams["layernorm"])
        
        #array for storing unpooling results
        self.unpool = []

    def forward(self, x, edge_index):
        input_x = x

        x = self.input_network(x)

        # Shortcut connect the inputs onto the hidden representation
        x = torch.cat([x, input_x], dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            x_inital = x

            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, edge_index))

            # Apply node network
            x = self.node_network(x, e, edge_index)

            # Shortcut connect the inputs onto the hidden representation
            x = torch.cat([x, input_x], dim=-1)

            # Residual connection
            x = x_inital + x

        edge_scores = self.edge_network(x, edge_index)
    
        input_x = x
        
        batch = torch.zeros(x.size()[0],dtype=torch.int64)
        
        x, edge_index, batch, edge_scores, unpool_info = self.edge_network.__merge_edges__(x,edge_index,batch,edge_scores)
        
        return edge_scores, unpool_info