# System imports
import os
import sys

# External imports
import torch
from tqdm import tqdm

import copy

sys.path.append("../..")
from LightningModules.SuperEmbedding.Models.undirected_embedding import UndirectedEmbedding
from LightningModules.SuperEmbedding.utils import build_edges, graph_intersection

device = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingInferenceBuilder:
    def __init__(self, model, output_dir, max_radius = 0.1, target_eff = None, overwrite=False):
        self.output_dir = output_dir
        self.model = model
        self.overwrite = overwrite
        self.max_radius = max_radius
        self.target_eff = target_eff

        # Prep the directory to produce inference data to
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

    def build(self):
        print("Training finished, running inference to build graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": self.model.trainset,
            "val": self.model.valset,
            "test": self.model.testset,
        }
        self.model.eval()
        with torch.no_grad():
            for datatype, dataset in datasets.items():
                for batch in tqdm(dataset, desc=f"Building graphs for {datatype}"):
                    if (
                        not os.path.exists(
                            os.path.join(
                                self.output_dir, datatype, batch.event_file[-4:]
                            )
                        )
                    ) or self.overwrite:
                        batch_to_save = copy.deepcopy(batch)
                        batch_to_save = batch_to_save.to(self.model.device)                            
                        self.construct_downstream(batch_to_save, datatype)


    def construct_downstream(self, batch, datatype):

        spatial = self.model(batch)

        # Make truth bidirectional
        e_bidir = torch.cat(
            [
                batch.modulewise_true_edges,
                batch.modulewise_true_edges.flip(0),
            ],
            axis=-1,
        )

        e_signal = torch.cat(
            [
                batch.signal_true_edges,
                batch.signal_true_edges.flip(0),
            ],
            axis=-1,
        )

        # Build the radius graph with initial radius
        edges = build_edges(
            spatial,
            spatial,
            indices=None,
            r_max=self.max_radius,
            k_max=1000,
        )

        _, y_signal = graph_intersection(edges, e_signal)
        print("Eff:", y_signal.sum() / e_signal.shape[1], "Pur:", y_signal.sum() / y_signal.shape[0])
        

        if self.target_eff is not None:
            # Build the radius graph with target efficiency
            distances = torch.pairwise_distance(spatial[e_bidir[0]], spatial[e_bidir[1]])
            distances, _ = torch.sort(distances, descending=False)
            target_radius = min(distances[int(len(distances) * (self.target_eff / 100))].item(), self.max_radius)
            d_sq = torch.sum((spatial[edges[0]] - spatial[edges[1]]) ** 2, axis=-1)
            edges = edges[:, d_sq < target_radius ** 2]

        # # Arbitrary ordering to remove half of the duplicate edges
        R_dist = batch.x[:, 0] ** 2 + batch.x[:, 2] ** 2
        edges = edges[:, (R_dist[edges[0]] <= R_dist[edges[1]])]

        edges, y_cluster = graph_intersection(edges, e_bidir)

        # Re-introduce random direction, to avoid training bias
        random_flip = torch.randint(2, (edges.shape[1],)).bool()
        edges[0, random_flip], edges[1, random_flip] = (
            edges[1, random_flip],
            edges[0, random_flip],
        )

        batch.edge_index = edges
        batch.y = y_cluster

        self.save_downstream(batch, datatype)

    def save_downstream(self, batch, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)


def main():

    checkpoint_path = "/global/cfs/cdirs/m3443/data/lightning_models/lightning_checkpoints/ITk_GravMetric_Study_E/1zhse1oi/checkpoints/last.ckpt"

    model = UndirectedEmbedding.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    output_dir = "/global/cfs/cdirs/m3443/data/ITk-upgrade/processed/embedding_processed/1GeV_cut_barrel"
    model.hparams["train_split"] = [800, 80, 10]
    model.setup(stage="fit")

    edge_builder = EmbeddingInferenceBuilder(model, output_dir, overwrite=True, max_radius=0.09)
    edge_builder.build()


if __name__ == "__main__":
    main()
