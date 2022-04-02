# import all
import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sps
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tracking_utils import *
from plotting_utils import *

def load_single_pytorch_file(file):
        """
        Loads a single Pytorch Geometric file
        """
        return torch.load(file, map_location="cpu")

class TrackingData():
    """
    A class that holds a list of Events, specifically for the tracking pipeline.
    An Event contains a Graph and an EventTruth object. 
    """
    def __init__(self, files):   
        self.files = files
        self.event_data = None
        self.events = None
        self.evaluation = None

        logging.info("Loading files")
        self.__load_files()
        assert self.event_data is not None # Test if files are loaded
        
        logging.info("Building events")
        self.__build_events()
        assert self.events is not None # Test if events are built

        
    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        return event

    def __load_files(self):
        """
        Loads files based on type
        """
        file_type = self.__get_file_type()

        if file_type == "pytorch_geometric":
            self.event_data = self.__load_pytorch_files()
        else:
            raise ValueError("Unknown file type")

    def __build_events(self):
        """
        Builds Event objects from event data
        """
        # self.events = []
        # for data in tqdm(self.event_data):
        #     self.events.append(Event(data))
        
        self.events = process_map(Event, self.event_data)#, max_workers=1)

    def __get_file_type(self):
        """
        Determine type of file
        """
        try:
            sample = torch.load(self.files[0])
            if str(type(sample)) == "<class 'torch_geometric.data.data.Data'>":
                return "pytorch_geometric"
            else:
                raise ValueError("Unknown file type, this is not a Pytorch Geometric file")
        except:
            raise ValueError("Unknown file type, there are still more file types to be added!")

    
    def __load_pytorch_files(self):
        """
        Loads all Pytorch geometric files in file list
        """
        # data = []
        # for file in tqdm(self.files):
        #     data.append(torch.load(file, map_location="cpu"))

        data = process_map(load_single_pytorch_file, self.files)#, max_workers=1)
        return data

    def build_candidates(self, building_method="CC", sanity_check=False, **kwargs):
        """
        Builds track candidates from events
        """
        logging.info(f"Building candidates with sanity check: {sanity_check}")   
        
        for event in tqdm(self.events):
            event.build_candidates(building_method, sanity_check, **kwargs)

    def evaluate_candidates(self, evaluation_method="matching", **kwargs):
        """
        Evaluates track candidates from events
        """
        logging.info("Evaluating candidates")
        for event in tqdm(self.events):
            event.evaluate_candidates(evaluation_method, **kwargs)
                
        n_true_tracks, n_reco_tracks, n_matched_particles, n_matched_tracks, n_duplicated_tracks, n_single_matched_particles = 0, 0, 0, 0, 0, 0
        for event in self.events:
            n_true_tracks += event.candidates.evaluation["n_true_tracks"]
            n_reco_tracks += event.candidates.evaluation["n_reco_tracks"]
            n_matched_particles += event.candidates.evaluation["n_matched_particles"]
            n_single_matched_particles += event.candidates.evaluation["n_single_matched_particles"]
            n_matched_tracks += event.candidates.evaluation["n_matched_tracks"]
            n_duplicated_tracks += event.candidates.evaluation["n_duplicated_tracks"]
            building_method = event.candidates.building_method

        self.evaluation = {
                "building_method": building_method,
                "evaluation_method": evaluation_method,
                "eff": n_matched_particles / n_true_tracks,
                "single_eff": n_single_matched_particles / n_true_tracks,
                "fr": 1 - (n_matched_tracks / n_reco_tracks),
                "dup": n_duplicated_tracks / n_reco_tracks,
            }

        print(self.evaluation)
        print(f"n_true_tracks: {n_true_tracks}, n_reco_tracks: {n_reco_tracks}, n_matched_particles: {n_matched_particles}, n_matched_tracks: {n_matched_tracks}, n_duplicated_tracks: {n_duplicated_tracks}")

    def plot_evaluation(self, metric="eff", observable="eta", **kwargs):
        """
        Plots evaluation of candidates
        """
        
        if self.evaluation is None:
            raise ValueError("No evaluation available")
        
        if self.evaluation["evaluation_method"] == "matching":
            self.__plot_matching_evaluation(metric, observable, **kwargs)
        else:
            raise NotImplementedError("Plotting not implemented yet for that method")

    def __plot_matching_evaluation(self, metric="eff", observable="eta", **kwargs):
        """
        Plots matching evaluation of candidates
        """
        all_particles = pd.concat([event.candidates.evaluation["particles"].merge(event.event_truth.particles, on="particle_id", how="inner") for event in self.events])

        plot_observable_performance(all_particles)


class Event():
    """
    An Event contains a Graph and an EventTruth object. It represents a unit of particle physics data.    
    """
    def __init__(self, data):
        
        self.graph = None
        self.event_truth = None
        self.candidates = None
        
        self.data = self.__process_data(data)
      
    def __process_data(self, data):
        """
        Processes data to be used in the pipeline
        """

        if str(type(data)) == "<class 'torch_geometric.data.data.Data'>":    
            self.graph = Graph(data_dict = data.to_dict())
            self.event_truth = EventTruth(event_file = data.event_file)
        else:
            raise ValueError("Unknown data type")

    # Define representation
    def __repr__(self):
        return f"Event(graph=({len(self.graph.hits['x'])} hits, {self.graph.edges['edge_index'].shape[1]} edges), event_truth=({len(self.event_truth)} particles), candidates=({len(self.candidates)} candidates))"

    def build_candidates(self, building_method="CC", sanity_check=False, **kwargs):
        """
        Builds track candidates from event
        """
        self.candidates = self.graph.build_candidates(building_method, sanity_check, **kwargs)

    def evaluate_candidates(self, method="matching", **kwargs):
        """
        Evaluates track candidates from event
        """

        self.candidates.evaluate(method, self.event_truth, **kwargs)
    

class Graph():
    def __init__(self, data_dict):
        self.hits = None 
        self.edges = None
        self.graph_data = None

        assert type(data_dict) == dict, "Data must be a dictionary"
        self.__process_data(data_dict)

        # Test if data is loaded
        assert self.hits is not None
        assert self.edges is not None
        assert self.graph_data is not None

    # Define representation
    def __repr__(self):
        return f"Graph(hits={self.hits}, edges={self.edges}, graph_data={self.graph_data})"
    
    def __len__(self):
        return len(self.hits["x"])

    def __process_data(self, data):
        """
        Processes data to be used in the pipeline
        """
        if type(data) == dict:
            self.__get_hit_data(data)
            self.__get_edge_data(data)
            self.__get_graph_data(data)
        else:
            raise ValueError("Unknown data type")

    def __get_hit_data(self, data):
        """
        Returns hit data
        """
        self.hits = {}

        assert "x" in data.keys(), "At least need a feature called x, otherwise define default node feature in config" # Check if x is in data

        for key in data.keys():
            if len(data[key]) == len(data["x"]):
                self.hits[key] = data[key]

    def __get_edge_data(self, data):
        """
        Returns edge data
        """
        self.edges = {}

        assert "edge_index" in data.keys(), "At least need a feature called edge_index, otherwise define default edge feature in config" # Check if edge_index is in data

        for key in data.keys():
            if (
                len(data[key].shape) > 1 and data[key].shape[1] == data["edge_index"].shape[1] or
                len(data[key].shape) == 1 and data[key].shape[0] == data["edge_index"].shape[1]
            ): 
                self.edges[key] = data[key]
    
    def __get_graph_data(self, data):
        """
        Returns graph data
        """
        self.graph_data = {k: data[k] for k in data.keys() - (self.hits.keys() & self.edges.keys())}   

    def build_candidates(self, building_method="CC", sanity_check=False, **kwargs):
        """
        Builds track candidates from graph
        """
    
        if building_method == "CC":
            candidates = self.__get_connected_components(sanity_check, **kwargs)
        elif building_method == "KF":
            candidates = self.__get_kf_candidates(**kwargs)
        else:
            raise ValueError("Unknown building method")

        return candidates

    def __get_connected_components(self, sanity_check=False, score_cut=0.5, **kwargs):
        """
        Builds connected components from graph
        """
        if sanity_check:
            edge_mask = self.edges["pid_signal"].bool()
        else:
            edge_mask = self.edges["scores"] > score_cut

        row, col = self.edges["edge_index"][:, edge_mask]
        edge_attr = np.ones(row.size(0))

        N = self.hits["x"].size(0)
        sparse_edges = sps.coo_matrix((edge_attr, (row.numpy(), col.numpy())), (N, N))

        num_candidates, candidate_labels = sps.csgraph.connected_components(sparse_edges, directed=False, return_labels=True)   
        
        candidates = Candidates(self.hits["hid"], candidate_labels, building_method="CC")
        return candidates

    def __get_kf_candidates(self, **kwargs):
        """
        Builds KF candidates from graph
        """
        raise NotImplementedError("KF candidates not implemented yet")


class EventTruth():
    def __init__(self, event_file):
        self.particles = None
        self.hit_truth = None 

        assert type(event_file) == str or type(event_file) == np.str_, "Event file must be a string"
        self.__process_data(event_file)

        # Test data loaded properly
        assert self.particles is not None
        assert self.hit_truth is not None

    # Define representation
    def __repr__(self):
        return f"EventTruth(particles={self.particles}, hit_truth={self.hit_truth})"
    
    def __len__(self):
        return len(self.particles)

    def __process_data(self, event_file):
        """
        Processes data to be used in the pipeline
        """
        self.__get_particle_data(event_file)
        self.__get_hit_truth_data(event_file)

    def __get_particle_data(self, event_file):
        """
        Returns particle data
        """

        try:
            particle_filename = event_file + "-particles.csv"
            self.particles = pd.read_csv(particle_filename)
        except:
            raise ValueError("Could not find particles file")


    def __get_hit_truth_data(self, event_file):
        """
        Returns hit truth data
        """
        
        try:
            hit_truth_filename = event_file + "-truth.csv"
            self.hit_truth = pd.read_csv(hit_truth_filename)
            self.hit_truth = self.__process_hit_truth(self.hit_truth)
        except:
            raise ValueError("Could not find hit truth file")

    def __process_hit_truth(self, hit_truth):
        """
        Processes hit truth data
        """
        hit_truth.drop_duplicates(subset=["hit_id"], inplace=True)

        return hit_truth


class Candidates():
    def __init__(self, hit_ids, track_ids, building_method, **kwargs):
        self.hit_ids = hit_ids
        self.track_ids = track_ids 
        self.building_method = building_method
        self.evaluation = None 

    def __repr__(self):
        return f"Candidates(hit_ids={self.hit_ids}, track_ids={self.track_ids}, evaluation={self.evaluation})"
    
    def __len__(self):
        return len(np.unique(self.track_ids))

    def get_df(self):
        """
        Returns dataframe of candidates
        """
        df = pd.DataFrame({"hit_id": self.hit_ids, "track_id": self.track_ids})
        return df

    def evaluate(self, method, event_truth, **kwargs):
        """
        Returns evaluation of candidates
        """

        if method == "matching":
            self.evaluation = self.__matching_reconstruction(event_truth.particles, event_truth.hit_truth, **kwargs)
        elif method == "iou":
            self.evaluation = self.__iou_reconstruction(**kwargs)
        else:
            raise ValueError("Unknown method")
    

    def __matching_reconstruction(self, particles, hit_truth, **kwargs):
        """
        Evaluates track candidates from event with matching criteria. Criteria given by ratios of common hits in candidates ("reconstructed") and particles ("truth")        
        """

        particles, candidates = match_reco_tracks(self.get_df(), hit_truth, particles, **kwargs)
        (n_true_tracks, n_reco_tracks, 
        n_matched_particles, n_single_matched_particles, n_matched_tracks, 
        n_duplicated_tracks, n_matched_tracks_poi) = get_statistics(particles, candidates)

        evaluation = {
            "evaluation_method": "matching",
            "particles": particles,
            "candidates": candidates,
            "eff": n_matched_particles / n_true_tracks,
            "fr": 1 - (n_matched_tracks / n_reco_tracks),
            "dup": n_duplicated_tracks / n_reco_tracks,
            "n_true_tracks": n_true_tracks, 
            "n_reco_tracks": n_reco_tracks,
            "n_matched_particles": n_matched_particles,
            "n_single_matched_particles": n_single_matched_particles,
            "n_matched_tracks": n_matched_tracks,
            "n_duplicated_tracks": n_duplicated_tracks,
            "n_matched_tracks_poi": n_matched_tracks_poi
        }

        return evaluation
        
    def __iou_reconstruction(self, **kwargs):
        """
        Evaluates track candidates from event with Intersection over Union (IoU)
        """
        raise NotImplementedError("IOU reconstruction not implemented yet")
