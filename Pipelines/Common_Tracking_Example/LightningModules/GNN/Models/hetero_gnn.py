import sys
from webbrowser import get

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
from itertools import combinations_with_replacement, product

from ..hetero_gnn_base import LargeGNNBase, PyGHeteroGNNBase
from ..utils import make_mlp, get_region

from .submodels.edge_decoders import HomoDecoder, HeteroDecoder, EdgeClassifier
from .submodels.convolutions import HomoConv, HeteroConv, InteractionHeteroConv, InteractionMessagePassing, PyGInteractionHeteroConv, NodeOnlyHeteroConv
from .submodels.encoders import HeteroEdgeEncoder, HomoEncoder, HeteroEncoder, HeteroNodeEncoder, HeteroEdgeConv, EdgeEncoder, EdgeUpdater
from torch_geometric.nn import GATv2Conv, HeteroConv as PyGHeteroConv


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

        self.node_encoders = self.make_node_encoder()
        self.edge_encoders = self.make_edge_encoder()
        self.convs, self.edge_updaters = self.make_conv()
        self.edge_classifiers = self.make_edge_classifier()
    
    def make_node_encoder(self, params={}):
        hparams = self.hparams.copy()
        for key, val in params.items():
            hparams[key] = val
        node_encoders = HeteroNodeEncoder(hparams)
        return node_encoders

    def make_edge_encoder(self, params={}):
        hparams = self.hparams.copy()
        for key, val in params.items():
            hparams[key] = val
        edge_encoders = HeteroEdgeEncoder(hparams)
        return  edge_encoders

    def make_conv(self, params={}):
        hparams = self.hparams.copy()
        for key, val in params.items():
            hparams[key] = val
        
        module_convs = torch.nn.ModuleList([])
        module_edge_updaters = torch.nn.ModuleList([])
        convs, edge_updaters= {}, {}
        for model0, model1 in combinations_with_replacement(hparams['model_ids'], r=2):
            conv = InteractionMessagePassing(hparams=hparams)
            convs[(get_region(model0), 'connected_to', get_region(model1))] = convs[(get_region(model1), 'connected_to', get_region(model0))] = conv

            edge_updater = EdgeUpdater(hparams=hparams)
            edge_updaters[(get_region(model0), 'connected_to', get_region(model1))] = edge_updaters[(get_region(model1), 'connected_to', get_region(model0))] = edge_updater
        
        conv = PyGInteractionHeteroConv(convs, aggr=hparams.get('modulewise_aggregation', 'mean'))
        edge_updater = HeteroEdgeConv(edge_updaters, aggr=hparams.get('modulewise_aggregation', 'mean'))
        for _ in range(hparams['n_graph_iters']):
            if hparams.get('recurrent'): 
                module_convs.append(conv)
                module_edge_updaters.append(edge_updater)
            else: 
                convs, edge_updaters= {}, {}
                for model0, model1 in combinations_with_replacement(hparams['model_ids'], r=2):
                    conv = InteractionMessagePassing(hparams=hparams)
                    convs[(get_region(model0), 'connected_to', get_region(model1))] = convs[(get_region(model1), 'connected_to', get_region(model0))] = conv

                    edge_updater = EdgeUpdater(hparams=hparams)
                    edge_updaters[(get_region(model0), 'connected_to', get_region(model1))] = edge_updaters[(get_region(model1), 'connected_to', get_region(model0))] = edge_updater
                
                conv = PyGInteractionHeteroConv(convs, aggr=hparams.get('modulewise_aggregation', 'mean'))
                edge_updater = HeteroEdgeConv(edge_updaters, aggr=hparams.get('modulewise_aggregation', 'mean'))
                module_convs.append(conv)
                module_edge_updaters.append(edge_updater)
        return module_convs, module_edge_updaters

    def make_edge_classifier(self, params={}):
        hparams = self.hparams.copy()
        for key, val in params.items():
            hparams[key] = val
        
        edge_classifiers = {}
        for model0, model1 in combinations_with_replacement(hparams['model_ids'], r=2):
            ec = EdgeClassifier(hparams)
            edge_classifiers[(get_region(model0), 'connected_to', get_region(model1))] = edge_classifiers[(get_region(model1), 'connected_to', get_region(model0))] = ec
        
        edge_classifiers = HeteroEdgeConv(edge_classifiers)

        return edge_classifiers
        
    def forward(self, batch):
        x_dict, edge_index_dict= batch.x_dict, batch.edge_index_dict
        
        x_dict = self.node_encoders(x_dict)
        edge_dict = self.edge_encoders(x_dict, edge_index_dict)

        for conv, edge_updater in zip(self.convs, self.edge_updaters):
            x_dict = conv(x_dict, edge_index_dict, edge_dict)
            edge_dict = edge_updater(x_dict, edge_index_dict, edge_dict)
        
        return self.edge_classifiers(x_dict, edge_index_dict, edge_dict)

class HybridHeteroGNN(PyGHeteroGNNBase):

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
        self.edge_encoders = HeteroEdgeEncoder(self.hparams)

        if not self.hparams.get("recurrent"):
            self.conv = nn.ModuleList([InteractionHeteroConv(hparams)]*self.hparams["n_graph_iters"])
        else:
            self.conv = nn.ModuleList([InteractionHeteroConv(hparams) for _ in range(self.hparams["n_graph_iters"])])

        if self.hparams['hetero_level'] < 4:
            edge_classifier = EdgeClassifier(self.hparams)
            self.edge_classifiers = HeteroEdgeConv({
                (get_region(model0), 'connected_to', get_region(model1)): edge_classifier
                for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
            })
        else:
            if self.hparams.get('hetero_reduce'):
                convs = {}
                for model0, model1 in combinations_with_replacement(self.hparams['model_ids'], r=2):
                    edge_classifier = EdgeClassifier(self.hparams)
                    convs[ (get_region(model0), 'connected_to', get_region(model1))] = convs[ (get_region(model1), 'connected_to', get_region(model0))] = edge_classifier 
                self.edge_classifiers = HeteroEdgeConv(convs)
            else:
                self.edge_classifiers = HeteroEdgeConv({
                    (get_region(model0), 'connected_to', get_region(model1)): EdgeClassifier(self.hparams)
                    for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
                })

    def forward(self, batch):
        x_dict, edge_index_dict= batch.x_dict, batch.edge_index_dict
        
        x_dict = self.node_encoders(x_dict)
        edge_dict = self.edge_encoders(x_dict, edge_index_dict)  

        for _, conv in enumerate(self.conv):

            x_dict, edge_dict = conv(x_dict, edge_index_dict, edge_dict)

        return self.edge_classifiers(x_dict, edge_index_dict, edge_dict)

class NodeOnlyHybridHeteroGNN(HybridHeteroGNN):
    def __init__(self, hparams: dict):

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
        edge_hparams = hparams.copy()
        edge_hparams['hetero_level']=0
        self.edge_encoders = HeteroEdgeEncoder(self.hparams)

        if not self.hparams.get("recurrent"):
            self.conv = nn.ModuleList([NodeOnlyHeteroConv(hparams)]*self.hparams["n_graph_iters"])
        else:
            self.conv = nn.ModuleList([NodeOnlyHeteroConv(hparams) for _ in range(self.hparams["n_graph_iters"])])

        edge_classifier = EdgeClassifier(self.hparams)
        self.edge_classifiers = HeteroEdgeConv({
            (get_region(model0), 'connected_to', get_region(model1)): edge_classifier
            for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
        })

class AttentionGNN(PyGHeteroGNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.node_encoders = HeteroNodeEncoder(self.hparams)

        if self.hparams.get("recurrent"):
            convs = {}
            common_conv = GATv2Conv(hparams['hidden'], hparams['hidden'], heads=self.hparams.get('attention_heads', 2), concat=False, share_weights=True)
            for model0, model1 in combinations_with_replacement(self.hparams['model_ids'], r=2):
                conv = common_conv if self.hparams['hetero_level'] < 3 else  GATv2Conv(hparams['hidden'], hparams['hidden'], heads=self.hparams.get('attention_heads', 2), concat=False, share_weights=True)
                convs[(get_region(model0), 'connected_to', get_region(model1))] = convs[(get_region(model1), 'connected_to', get_region(model0))] = conv
            self.conv = nn.ModuleList([PyGHeteroConv(convs)]*self.hparams["n_graph_iters"])
        else:
            module_list = []
            for _ in range(self.hparams['n_graph_iters']):
                convs = {}
                common_conv = GATv2Conv(hparams['hidden'], hparams['hidden'], heads=self.hparams.get('attention_heads', 2), concat=False, share_weights=True)
                for model0, model1 in combinations_with_replacement(self.hparams['model_ids'], r=2):
                    conv = common_conv if self.hparams['hetero_level'] < 3 else  GATv2Conv(hparams['hidden'], hparams['hidden'], heads=self.hparams.get('attention_heads', 2), concat=False, share_weights=True)
                    convs[(get_region(model0), 'connected_to', get_region(model1))] = convs[(get_region(model1), 'connected_to', get_region(model0))] = conv
                module_list.append(
                    PyGHeteroConv(convs)
                )
            self.conv = nn.ModuleList(module_list)

        self.edge_encoders = HeteroEdgeEncoder(self.hparams)

        if self.hparams['hetero_level'] < 4:
            edge_classifier = EdgeClassifier(self.hparams)
            self.edge_classifiers = HeteroEdgeConv({
                (get_region(model0), 'connected_to', get_region(model1)): edge_classifier
                for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids'])
            })
        else:
            convs = {}
            for model0, model1 in combinations_with_replacement(self.hparams['model_ids'], r=2):
                edge_classifier = EdgeClassifier(self.hparams)
                convs[ (get_region(model0), 'connected_to', get_region(model1))] = convs[ (get_region(model1), 'connected_to', get_region(model0))] = edge_classifier 
            self.edge_classifiers = HeteroEdgeConv(convs)
            

    def forward(self, batch):
        x_dict, edge_index_dict= batch.x_dict, batch.edge_index_dict
        
        x_dict = self.node_encoders(x_dict)
        
        for conv in self.conv:
            x_dict = conv(x_dict, edge_index_dict)
        
        edge_dict = self.edge_encoders(x_dict, edge_index_dict)
        return self.edge_classifiers(x_dict, edge_index_dict, edge_dict)