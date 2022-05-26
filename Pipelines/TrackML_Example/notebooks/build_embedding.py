# System imports
import os

# External imports
import numpy as np
import random
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingInferenceBuilder:
    def __init__(self, model, split = [80, 10, 10], overwrite=False, knn_max = 1000, radius = 0.1):
        
        self.model = model
        self.output_dir = self.model.hparams["output_dir"]
        self.input_dir = self.model.hparams["input_dir"]
        self.overwrite = overwrite
        self.split = split
        self.knn_max = knn_max
        self.radius = radius
        
        single_file_split = [1, 1, 1]
        model.hparams["train_split"] = single_file_split
        model.setup(stage="fit")
        

    def build(self):
        print("Training finished, running inference to build graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = self.prepare_datastructure()
        
        self.model.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for event in tqdm(dataset):
                    
                    event_file = os.path.join(self.input_dir, event)
                    if (
                        not os.path.exists(
                            os.path.join(
                                self.output_dir, datatype, event_file[-4:]
                            )
                        )
                    ) or self.overwrite:
                        batch = torch.load(event_file).to(self.model.device)
                        self.construct_downstream(batch, datatype)

    def prepare_datastructure(self):
        # Prep the directory to produce inference data to
        self.output_dir = self.model.hparams.output_dir
        self.datatypes = ["train", "val", "test"]

        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

        all_events = os.listdir(self.model.hparams["input_dir"])
        random.shuffle(all_events)
        self.dataset_list = np.split(np.array(all_events), np.cumsum(self.split))
        
        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": list(self.dataset_list[0]),
            "val": list(self.dataset_list[1]),
            "test": list(self.dataset_list[2]),
        }
        
        return datasets
                        
    def construct_downstream(self, batch, datatype):

        batch = self.select_data(batch)
        
        y_cluster, e_spatial, e_bidir = self.get_performance(
            self.model, batch, r_max= self.radius, k_max=self.knn_max
        )
        
        module_mask = batch.modules[e_spatial[0]] != batch.modules[e_spatial[1]]
        y_cluster, e_spatial = y_cluster[module_mask], e_spatial[:, module_mask]
        
        # Arbitrary ordering to remove half of the duplicate edges
        R_dist = torch.sqrt(batch.x[:, 0] ** 2 + batch.x[:, 2] ** 2)
        e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]

        e_spatial, y_cluster = self.model.get_truth(batch, e_spatial, e_bidir)

        # Re-introduce random direction, to avoid training bias
        random_flip = torch.randint(2, (e_spatial.shape[1],)).bool()
        e_spatial[0, random_flip], e_spatial[1, random_flip] = (
            e_spatial[1, random_flip],
            e_spatial[0, random_flip],
        )

        batch.edge_index = e_spatial
        batch.y = y_cluster

        self.save_downstream(batch, datatype)

    def get_performance(self, batch, r_max, k_max):
        with torch.no_grad():
            results = self.model.shared_evaluation(batch, 0, r_max, k_max)

        return results["truth"], results["preds"], results["truth_graph"]

    def save_downstream(self, batch, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)
            
    def select_data(self, event):
    
        event.signal_true_edges = event.modulewise_true_edges
        
        if (
            ("pt" in event.keys and self.model.hparams["pt_signal_cut"] > 0)
        ):
            edge_subset = (
                (event.pt[event.signal_true_edges] > self.model.hparams["pt_signal_cut"]).all(0)
            )

            event.signal_true_edges = event.signal_true_edges[:, edge_subset]
        
        return event