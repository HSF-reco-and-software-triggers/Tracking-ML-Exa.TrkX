# System imports
import sys
import os

# 3rd party imports
# import pytorch_lightning as pl
# from pytorch_lightning import LightningModule
# import torch
# from torch.nn import Linear
# from torch.utils.data import random_split
# from torch_geometric.data import DataLoader
# from torch_cluster import radius_graph
# import numpy as np


# Local imports
# from .utils import graph_intersection
# if torch.cuda.is_available():
#     from .utils import build_edges, res
#     device = 'cuda'
# else:
#     device = 'cpu'

# def load_datasets(input_dir, train_split, seed = 0):
#     '''
#     Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
#     changed across final varied runs, but can be left as default for experimentation.
#     '''
#     torch.manual_seed(seed)
#     all_events = os.listdir(input_dir)
#     all_events = sorted([os.path.join(input_dir, event) for event in all_events])
#     loaded_events = [torch.load(event) for event in all_events[:sum(train_split)]]
#     train_events, val_events, test_events = random_split(loaded_events, train_split)
#
#     return train_events, val_events, test_events

class Bar:
    """
    A test class
    """
    def __init__(self):
        self.hello = "Hi there"

    def test_method(self, new_hello):
        """
        A test method
        """
        self.hello = new_hello
