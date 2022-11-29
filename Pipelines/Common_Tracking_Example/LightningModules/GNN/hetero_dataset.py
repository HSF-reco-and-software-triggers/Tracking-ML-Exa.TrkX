import torch 
import yaml, os
import matplotlib, seaborn as sns
from torch_geometric.data import HeteroData, Dataset
from pytorch_lightning import LightningModule
from itertools import combinations_with_replacement, product
from typing import Dict, Optional
from torch import Tensor
from torch_geometric.typing import Adj, EdgeType, NodeType

from .utils import process_data, get_region, get_hetero_data, process_hetero_data
import random
from functools import partial

class LargeHeteroDataset(Dataset):
    def __init__(self, root, subdir, hparams, num_events=-1, process_function=None):
        self.subdir = os.path.join(root, subdir)
        super().__init__(self.subdir, transform=None, pre_transform=None, pre_filter=None)
        
        self.hparams = hparams
        self.process_fuction = process_function
        
        self.input_paths = os.listdir(self.subdir)
        if "sorted_events" in hparams.keys() and hparams["sorted_events"]:
            self.input_paths = sorted(self.input_paths)
        else:
            random.shuffle(self.input_paths)
        
        self.input_paths = [os.path.join(self.subdir, event) for event in self.input_paths][:num_events]
        
    def len(self):
        return len(self.input_paths)

    def get(self, idx):
        event = torch.load(self.input_paths[idx], map_location=torch.device('cpu'))
        event = process_hetero_data(event, self.hparams['pt_background_cut'], self.hparams['pt_signal_cut'], self.hparams['noise'], triplets=False, input_cut=None, handle_directed=self.hparams.get('handle_directed', False))
        hetero_data = get_hetero_data(event, self.hparams)
        
        return hetero_data