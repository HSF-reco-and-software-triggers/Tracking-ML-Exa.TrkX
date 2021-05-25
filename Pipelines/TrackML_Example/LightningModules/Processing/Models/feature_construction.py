# System imports
import sys
import os
import multiprocessing as mp
from functools import partial

# 3rd party imports
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.nn import Linear
import torch.nn as nn
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# Local imports
from ..utils.event_utils import prepare_event
from ..utils.detector_utils import load_detector


class FeatureStore(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.input_dir = self.hparams["input_dir"]
        self.output_dir = self.hparams["output_dir"]
        self.detector_path = self.hparams["detector_path"]
        self.n_files = self.hparams["n_files"]

        self.n_tasks = self.hparams["n_tasks"]
        self.task = 0 if "task" not in self.hparams else self.hparams["task"]
        self.n_workers = (
            self.hparams["n_workers"]
            if "n_workers" in self.hparams
            else len(os.sched_getaffinity(0))
        )
        self.build_weights = (
            self.hparams["build_weights"] if "build_weights" in self.hparams else True
        )
        self.show_progress = (
            self.hparams["show_progress"] if "show_progress" in self.hparams else True
        )

    def prepare_data(self):
        # Find the input files
        all_files = os.listdir(self.input_dir)
        all_events = sorted(
            np.unique([os.path.join(self.input_dir, event[:14]) for event in all_files])
        )[: self.n_files]

        # Split the input files by number of tasks and select my chunk only
        all_events = np.array_split(all_events, self.n_tasks)[self.task]

        # Define the cell features to be added to the dataset

        cell_features = [
            "cell_count",
            "cell_val",
            "leta",
            "lphi",
            "lx",
            "ly",
            "lz",
            "geta",
            "gphi",
        ]
        detector_orig, detector_proc = load_detector(self.detector_path)

        # Prepare output
        # output_dir = os.path.expandvars(self.output_dir) FIGURE OUT HOW TO USE THIS!
        os.makedirs(self.output_dir, exist_ok=True)
        print("Writing outputs to " + self.output_dir)

        # Process input files with a worker pool and progress bar
        process_func = partial(
            prepare_event,
            detector_orig=detector_orig,
            detector_proc=detector_proc,
            cell_features=cell_features,
            **self.hparams
        )
        process_map(process_func, all_events, max_workers=self.n_workers)
