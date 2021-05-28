from collections import namedtuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add,scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import softmax
from .utils import make_mlp

class EdgePooling(torch.nn.Module):
    r"""
    Adapted from Pytorch geometric library
    The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.
    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.
    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.
    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.
    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster", "batch", "new_edge_score"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0,
                 add_to_edge_score=0.5):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = make_mlp(2 * in_channels,[2 * in_channels] * 8 + [1])

        self.lin.apply(self.reset_parameters)

    def reset_parameters(self,m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            m.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.
        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score
       
        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info, e

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
        
        while i < 10 and ratio > 0.05:    
            #get max edge score for each node and edge index where it occurs
            max_score_0, max_indices_0 = scatter_max(edge_score, edge_index[0], dim=0, dim_size=x.shape[0])
            max_score_1, max_indices_1 = scatter_max(edge_score, edge_index[1], dim=0, dim_size=x.shape[0])

            #stack scores for each direction
            stacked_score, stacked_indices = torch.stack([max_score_0, max_score_1]), torch.stack([max_indices_0, max_indices_1]).T
            top_score , _ = torch.max(stacked_score, dim=0)
            top_score = top_score.to(device)

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
        
        #count the number of occurences of each node to find duplicates
        _, counts = torch.unique(new_node_index_map[0], return_counts=True)
        duplicates = torch.where(counts >= 2)
        cluster_mask = torch.ones_like(cluster,dtype=torch.bool)
        #if it finds a duplicate node, remove it from the cluster map
        if duplicates[0].size(0) > 0:
            d = duplicates[0]
            print(d.size(0))
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
            clusters_to_remove,_ = torch.sort(clusters_to_remove,descending=True)
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
        
        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.
        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.
        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.
        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """

        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)