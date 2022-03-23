# import all
import os
from sre_constants import NOT_LITERAL_UNI_IGNORE
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sps
from tqdm import tqdm

def get_statistics(particles, candidates):
    """
    Returns statistics of particles
    """

    fiducial_particles = particles[particles.is_fiducial]

    # n_true_tracks = particles.is_trackable.sum()
    n_true_tracks = (fiducial_particles.is_trackable).sum()
    n_reco_tracks = len(candidates.track_id.unique())
    n_single_matched_particles = (fiducial_particles.is_matched & particles.is_trackable).sum()
    n_matched_particles = (fiducial_particles.is_double_matched & particles.is_trackable).sum()
    n_matched_tracks = len(candidates[candidates.is_matched].track_id.unique())
    n_matched_tracks_poi = len(candidates[(candidates.is_matched & candidates.particle_id.isin(fiducial_particles.particle_id))].track_id.unique())
    n_duplicated_tracks = n_matched_tracks_poi - n_matched_particles

    return (n_true_tracks, n_reco_tracks, 
            n_matched_particles, n_single_matched_particles, n_matched_tracks, 
            n_duplicated_tracks, n_matched_tracks_poi)

def match_reco_tracks(
        reconstructed: pd.DataFrame,
        truth: pd.DataFrame, 
        particles: pd.DataFrame,
        min_hits_truth=9, min_hits_reco=5,
        min_pt=1000., max_eta=4., frac_reco_matched=0.5, frac_truth_matched=0.5, **kwargs):
    
    """    
    Args:
        truth: a dataframe with columns of ['hit_id', 'particle_id']
        reconstructed: a dataframe with columns of ['hit_id', 'track_id']
        particles: a dataframe with columns of 
            ['particle_id', 'pt', 'eta', 'radius', 'vz'].
            where radius = sqrt(vx**2 + vy**2) and 
            ['vx', 'vy', 'vz'] are the production vertex of the particle
        min_hits_truth: minimum number of hits for truth tracks
        min_hits_reco:  minimum number of hits for reconstructed tracks

    Returns:
        A tuple of (
            n_true_tracks: int, number of true tracks
            n_reco_tracks: int, number of reconstructed tracks
            n_matched_reco_tracks: int, number of reconstructed tracks
                matched to true tracks
            matched_pids: np.narray, a list of particle IDs matched
                by reconstructed tracks
        )
    """
    # 1. Filter truth tracks
    truth, particles = filter_truth(truth, particles, min_pt, max_eta)

    # 2. Get candidate track lengths
    reconstructed, candidate_lengths = get_candidate_lengths(reconstructed, min_hits_reco)
    
    # 3. Get true particle track lengths
    truth, particles, particle_lengths = get_particle_lengths(truth, particles, min_hits_truth)

    # 4. Get common and shared hits between candidates and particles
    reconstructed, reco_matching, truth_matching = get_shared_hits(truth, reconstructed, candidate_lengths, particle_lengths)

    # 5. Calculate matching fractions
    reco_matching, truth_matching = get_matching_fractions(reco_matching, truth_matching)
    
    # 6. Apply matching criteria
    particles, reconstructed = apply_matching_criteria(particles, reconstructed, reco_matching, truth_matching, frac_reco_matched, frac_truth_matched, **kwargs)

    return particles[["particle_id", "is_fiducial", "is_trackable", "is_matched", "is_double_matched"]], reconstructed

def filter_truth(truth, particles,  min_pt=1000., max_eta=4.0):
    # just in case particle_id == 0 included in truth.
    if 'particle_id' in truth.columns:
        truth = truth[truth.particle_id > 0]
    
    # TODO: Add these fiducial cuts to the config file
    fiducial_cut = (particles.status == 1) & (particles.barcode < 200000) & (particles.radius < 260) & (particles.charge.abs() > 0)
    fiducial_cut &= (particles.pt > min_pt) & (particles.eta < max_eta)
    
    # Make fiducial property of particles
    particles["is_fiducial"] = fiducial_cut.values
    # particles = particles[(particles.pt > min_pt) & (particles.eta.abs() < max_eta)]

    
    return truth, particles

def get_candidate_lengths(reconstructed, min_hits_reco=3):

    # get number of spacepoints in each reconstructed tracks
    candidate_lengths = reconstructed.track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_reco_hits"})

    # only tracks with a minimum number of spacepoints are considered
    candidate_lengths = candidate_lengths[candidate_lengths.n_reco_hits >= min_hits_reco]
    reconstructed = reconstructed[reconstructed.track_id.isin(candidate_lengths.track_id.values)]

    return reconstructed, candidate_lengths

def get_particle_lengths(truth, particles, min_hits_truth=3):

    # get number of spacepoints in each particle
    truth = truth.merge(particles, on='particle_id', how='left')
    particle_lengths = truth.particle_id.value_counts(sort=False).reset_index().rename(
        columns={"index":"particle_id", "particle_id": "n_true_hits"})
    
    # only particles leaves at least min_hits_truth spacepoints 
    # and with pT >= min_pt are considered.
    particles = particles.merge(particle_lengths, on=['particle_id'], how='left')

    is_trackable = particles.n_true_hits >= min_hits_truth
    particles = particles.assign(is_trackable=is_trackable)

    return truth, particles, particle_lengths

def get_shared_hits(truth, reconstructed, candidate_lengths, particle_lengths):

    # reconstructed has 3 columns [track_id, particle_id, hit_id]
    reconstructed = pd.merge(reconstructed, truth, on=['hit_id'], how='left')
    
    # n_common_hits and n_shared should be exactly the same 
    # for a specific track id and particle id

    # Each track_id will be assigned to multiple particles.
    # To determine which particle the track candidate is matched to, 
    # we use the particle id that yields a maximum value of n_common_hits / candidate_lengths,
    # which means the majority of the spacepoints associated with the reconstructed
    # track candidate comes from that true track.
    # However, the other way may not be true.
    reco_matching = reconstructed.groupby(['track_id', 'particle_id']).size()\
        .reset_index().rename(columns={0:"n_common_hits"})

    
    # Each particle will be assigned to multiple reconstructed tracks
    truth_matching = reconstructed.groupby(['particle_id', 'track_id']).size()\
        .reset_index().rename(columns={0:"n_shared"})

    # add number of hits to each of the maching dataframe
    reco_matching = reco_matching.merge(candidate_lengths, on=['track_id'], how='left')
    truth_matching = truth_matching.merge(particle_lengths, on=['particle_id'], how='left')
    
    return reconstructed, reco_matching, truth_matching

def get_matching_fractions(reco_matching, truth_matching):

    # calculate matching fraction
    reco_matching = reco_matching.assign(
        purity_reco=np.true_divide(reco_matching.n_common_hits, reco_matching.n_reco_hits))
    truth_matching = truth_matching.assign(
        purity_true = np.true_divide(truth_matching.n_shared, truth_matching.n_true_hits))

    # select the best match
    reco_matching['purity_reco_max'] = reco_matching.groupby(
        "track_id")['purity_reco'].transform(max)
    truth_matching['purity_true_max'] = truth_matching.groupby(
        "track_id")['purity_true'].transform(max)

    return reco_matching, truth_matching

def apply_matching_criteria(particles, reconstructed, reco_matching, truth_matching, frac_reco_matched=0.5, frac_truth_matched=0.5):

    matched_reco_tracks = reco_matching[
        (reco_matching.purity_reco_max > frac_reco_matched)
    ]
    matched_true_particles = truth_matching[
        (truth_matching.purity_true_max > frac_truth_matched) \
        & (truth_matching.purity_true == truth_matching.purity_true_max)]

    # Without double matching, we use one-directional criteria
    combined_match = matched_true_particles.merge(
        matched_reco_tracks, on=['track_id', 'particle_id'], how='inner')

    particles["is_matched"] = particles.particle_id.isin(matched_true_particles.particle_id).values
    particles["is_double_matched"] = particles.particle_id.isin(combined_match.particle_id).values
    reconstructed["is_matched"] = reconstructed.track_id.isin(matched_reco_tracks.track_id).values

    return particles, reconstructed