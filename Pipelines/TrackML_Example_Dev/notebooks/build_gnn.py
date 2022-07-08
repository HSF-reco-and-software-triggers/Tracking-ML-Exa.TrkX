# System imports
import os

# External imports
import torch
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class GNNInferenceBuilder:
    def __init__(self, model, overwrite=None):
        self.model = model
        self.output_dir = model.hparams["output_dir"]
        if overwrite is not None: 
            self.overwrite = overwrite
        elif "overwrite" in model.hparams:
            self.overwrite = model.hparams["overwrite"]
        else:
            self.overwrite = False

        # Prep the directory to produce inference data to
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

    def infer(self):
        print("Training finished, running inference to filter graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": self.model.trainset,
            "val": self.model.valset,
            "test": self.model.testset,
        }

        self.model.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                print(f"Building {datatype}")
                for batch in tqdm(dataset):
                    if (
                        not os.path.exists(
                            os.path.join(
                                self.output_dir, datatype, batch.event_file[-4:]
                            )
                        )
                    ) or self.overwrite:
                        batch = batch.to(self.model.device)
                        batch = self.construct_downstream(batch)
                        self.save_downstream(batch, datatype)


    def construct_downstream(self, batch):

        output = self.model.shared_evaluation(batch, 0, log=False)
        
        batch.scores = output["score"][: int(len(output["score"]) / 2)]

        return batch

    def save_downstream(self, batch, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)