import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
from itertools import product

from ..hetero_gnn_base import LargeGNNBase, PyGHeteroGNNBase
from ..utils import make_mlp, get_region

from .submodels.edge_decoders import HomoDecoder, HeteroDecoder
from .submodels.convolutions import HomoConv, HeteroConv, InteractionHeteroConv, InteractionMessagePassing
from .submodels.encoders import HeteroEdgeEncoder, HomoEncoder, HeteroEncoder, HeteroNodeEncoder, HeteroEdgeConv, EdgeEncoder, EdgeClassifier

class HeteroGNN(LargeGNNBase):

    """
    A heterogeneous interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        # Make input network
        if self.hparams["hetero_level"] >= 1:
            self.encoder = HeteroEncoder(hparams)
        else:
            self.encoder = HomoEncoder(hparams)

        # Make message passing network with the following configuration:
        # Check whether the MP network should be homogeneous or heterogeneous (i.e. hetero_level >= 2)
        # Check whether the MP network should share weights between iterations (i.e. is recurrent)
        if "hetero_level" in self.hparams and self.hparams["hetero_level"] >= 2:
            if "recurrent" in self.hparams and not self.hparams["recurrent"]:
                self.conv = nn.ModuleList([HeteroConv(hparams) for _ in range(self.hparams["n_graph_iters"])])
            else:
                self.conv = nn.ModuleList([HeteroConv(hparams)]*self.hparams["n_graph_iters"])
        else:
            if "recurrent" in self.hparams and not self.hparams["recurrent"]:
                self.conv = nn.ModuleList([HomoConv(hparams) for _ in range(self.hparams["n_graph_iters"])])
            else:
                self.conv = nn.ModuleList([HomoConv(hparams)]*self.hparams["n_graph_iters"])        

        # Make output network
        if self.hparams["hetero_level"] >= 3:
            self.decoder = HeteroDecoder(hparams)
        else:
            self.decoder = HomoDecoder(hparams)

    def concat_output(self, concat_nodes, concat_edges):
        encoded_nodes = torch.cat(concat_nodes, dim=-1)
        encoded_edges = torch.cat(concat_edges, dim=-1)
        
        return encoded_nodes, encoded_edges
    
    def forward(self, x, edge_index, volume_id):

        # Encode the graph features into the hidden space
        x.requires_grad = True
        encoded_nodes, encoded_edges = self.encoder(x, edge_index, volume_id)
        
        concat_nodes, concat_edges = [], []

        # Loop over iterations of edge and node networks
        for step in range(self.hparams["n_graph_iters"]):

            # encoded_nodes, encoded_edges = checkpoint(self.conv[step], encoded_nodes, edge_index, encoded_edges, volume_id)
            encoded_nodes, encoded_edges = self.conv[step](encoded_nodes, edge_index, encoded_edges, volume_id)
            # print(encoded_nodes.shape, encoded_edges.shape)
            
            if "concat_output" in self.hparams and self.hparams["concat_output"]:
                concat_nodes.append(encoded_nodes)
                concat_edges.append(encoded_edges)
        
        if "concat_output" in self.hparams and self.hparams["concat_output"]:
            # encoded_nodes, encoded_edges = checkpoint(self.concat_output, concat_nodes, concat_edges)
            encoded_nodes, encoded_edges = self.concat_output(concat_nodes, concat_edges)
            
        # Compute final edge scores
        # TODO: Apply output to SUM of directional edge features (across both directions!)
        return self.decoder(encoded_nodes, edge_index, encoded_edges, volume_id)

class PyGHeteroGNN(PyGHeteroGNNBase):

    def __init__(self, hparams):

        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        self.node_encoders = HeteroNodeEncoder(self.hparams)
        # self.edge_encoders = HeteroEdgeConv({
        #     (get_region(model0), 'connected_to', get_region(model1)): EdgeEncoder(self.hparams)
        #     for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
        # })
        self.edge_encoders = HeteroEdgeEncoder(self.hparams)

        # self.convs = torch.nn.ModuleList([

        # ])

        # conv = InteractionHeteroConv({
        #         (get_region(model0), 'connected_to', get_region(model1)): InteractionMessagePassing(hparams=self.hparams)
        #         for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
        #     }, aggr=self.hparams.get('modulewise_aggregation', 'mean'))
        # for _ in range(self.hparams['n_graph_iters']):
        #     if self.hparams.get('recurrent'): self.convs.append(conv)
        #     else: 
        #         conv = InteractionHeteroConv({
        #             (get_region(model0), 'connected_to', get_region(model1)): InteractionMessagePassing(hparams=self.hparams)
        #             for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
        #         }, aggr=self.hparams.get('modulewise_aggregation', 'mean'))
        #         self.convs.append(conv)

        if not self.hparams.get("recurrent"):
            self.conv = nn.ModuleList([InteractionHeteroConv(hparams)]*self.hparams["n_graph_iters"])
        else:
            self.conv = nn.ModuleList([InteractionHeteroConv(hparams) for _ in range(self.hparams["n_graph_iters"])])

        self.edge_classifiers = HeteroEdgeConv({
            (get_region(model0), 'connected_to', get_region(model1)): EdgeClassifier(self.hparams)
            for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
        })

        # self.decoder = HeteroDecoder(hparams)

    def forward(self, batch):
        x_dict, edge_index_dict= batch.x_dict, batch.edge_index_dict
        
        # for region in x_dict: x_dict[region].require_grad = True
        x_dict = self.node_encoders(x_dict)
        edge_dict = self.edge_encoders(x_dict, edge_index_dict)
        # for node_type in x_dict:
        #     batch[node_type].x = x_dict[node_type]
        # for edge_type in edge_dict:
        #     batch[edge_type].edge = edge_dict[edge_type]

        # batch = batch.to_homogeneous()

        # x_dict = checkpoint(self.node_encoders, x_dict) 
        # edge_dict = checkpoint(self.edge_encoders, x_dict, edge_index_dict)      

        for step, conv in enumerate(self.conv):
            # x_dict = checkpoint(conv, x_dict, edge_index_dict, edge_dict)
            # edge_dict = checkpoint(conv.edge_forward, x_dict, edge_index_dict, edge_dict)

            x_dict, edge_dict = conv(x_dict, edge_index_dict, edge_dict)
            # edge_dict = conv.edge_forward(x_dict, edge_index_dict, edge_dict=edge_dict)
        
        # batch = batch.to_heterogeneous(
        #     node_type=batch.node_type,
        #     edge_type=batch.edge_type,
        #     node_type_names=[ get_region(model) for model in self.hparams['model_ids'] ],
        #     edge_type_names=[(get_region(model0), 'connected_to', get_region(model1)) for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])]
        # )
        # print(batch.x, batch.edge_index, batch.edge, batch.volume_id)
        # return self.decoder(batch.x, batch.edge_index, batch.edge, batch.volume_id ).squeeze()  
        return self.edge_classifiers(x_dict, edge_index_dict, edge_dict)
        # return checkpoint(self.edge_classifiers, x_dict, edge_index_dict, edge_dict)
        # return self.node_encoders(x_dict)


