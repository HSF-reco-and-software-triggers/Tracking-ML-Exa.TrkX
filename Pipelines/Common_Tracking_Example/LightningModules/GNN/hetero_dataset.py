import torch 
import yaml, os
import matplotlib, seaborn as sns
from torch_geometric.data import HeteroData, Dataset
from pytorch_lightning import LightningModule
from itertools import combinations_with_replacement, product
from typing import Dict, Optional
from torch import Tensor
from torch_geometric.typing import Adj, EdgeType, NodeType

from .utils import process_data, get_region
import random
from functools import partial

class LargeHeteroDataset(Dataset):
    def __init__(self, root, subdir, hparams, num_events=-1, process_function=None):
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        
        self.subdir = subdir
        self.hparams = hparams
        self.process_fuction = process_function
        # if transform is not None:
        #     from functools import partial
        #     self.transform = partial(transform, pt_background_cut=self.hparams['pt_background_cut'], pt_signal_cut=self.hparams['pt_signal_cut'], noise=self.hparams['noise'], triplets=False, input_cut=None)
        
        self.input_paths = os.listdir(os.path.join(root, subdir))
        if "sorted_events" in hparams.keys() and hparams["sorted_events"]:
            self.input_paths = sorted(self.input_paths)
        else:
            random.shuffle(self.input_paths)
        
        self.input_paths = [os.path.join(root, subdir, event) for event in self.input_paths][:num_events]
        
    def len(self):
        return len(self.input_paths)
    
    def get(self, idx):
        event = torch.load(self.input_paths[idx], map_location=torch.device("cpu"))   
        if self.process_fuction is not None:
            event = self.process_fuction(event)  

        # creates a map array 
        map = torch.zeros_like(event.hid)
        for model in self.hparams['model_ids']:
            # map each 
            volume_id = model['volume_ids']
            homo_ids = event.hid[ torch.isin( event.volume_id, torch.tensor(volume_id) ) ]
            map[homo_ids] = torch.arange(homo_ids.shape[0])
        

        data = HeteroData()
        for model in self.hparams['model_ids']:
            region = get_region(model)
            mask = torch.isin( event.volume_id, torch.tensor(model['volume_ids']) )
            for attr in ['x', 'cell_data', 'pid', 'hid', 'pt', 'primary', 'nhits', 'modules', 'volume_id']:
                data[region][attr] = event[attr][mask]
            data[region]['mask'] = mask
        
        for model0, model1 in product(self.hparams['model_ids'], self.hparams['model_ids']):
            # ids = torch.tensor([model1['volume_ids'], model2['volume_ids']])
            id0, id1 = torch.tensor([model0['volume_ids']]), torch.tensor([model1['volume_ids']])
            region0, region1 = get_region(model0), get_region(model1)
            mask0 = torch.isin(event.volume_id[event.edge_index[0]], id0) 
            mask1 = torch.isin(event.volume_id[event.edge_index[1]], id1)
            mask = mask1 & mask0 #+ torch.isin(event.volume_id[event.edge_index[0]], id2) * torch.isin(event.volume_id[event.edge_index[1]],id1)
            edge_index = event.edge_index.T[mask].T
            edge_index = map[edge_index]
            data[region0, 'connected_to', region1].edge_index = edge_index
            data[region0, 'connected_to', region1].y = event.y[mask]
            # data[region0, 'connected_to', region1].y_pid = event.y_pid[mask]
            data[region0, 'connected_to', region1].truth = event[self.hparams['truth_key']][mask]
            for key in ['modulewise_true_edges', 'signal_true_edges']:
                mask = torch.isin(event.volume_id[event[key][0]], id0) * torch.isin(event.volume_id[event[key][1]], id1) #+ torch.isin(event.volume_id[event[truth_edge][0]], id2) * torch.isin(event.volume_id[event[truth_edge][1]], id1)
                data[region0, 'connected_to', region1][key] = event[key].T[mask].T
        return data 