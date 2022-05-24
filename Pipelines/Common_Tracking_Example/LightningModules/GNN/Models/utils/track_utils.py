import os
import numpy as np
import pandas as pd

def get_tracking_metrics():
    pass

def get_metrics_matrices(
    edge_scores,
    particles_df,
    truth_df,
    edge_cut_range=None,
    track_length_range=None,
    ROI=None,
):

    if edge_cut_range is None:
        edge_cut_range = np.arange(0.1, 0.9, 0.1)
    if track_length_range is None:
        track_length_range = np.arange(1, 12)
    if ROI is None:
        ROI = {"pt": {"min": 1}}

    efficiency_matrix = np.zero((len(edge_cut_range), len(track_length_range)))
    fake_rate_matrix = np.zero((len(edge_cut_range), len(track_length_range)))

    for i, edge_cut in enumerate(edge_cut_range):
        for j, track_length in enumerate(track_length_range):

            track_candidates = get_track_candidates(edge_scores, edge_cut)
            # TODO
            # track_candidates = clean_track_candidates(track_candidates)

            selected_tracks = get_selected_tracks(track_candidates, track_length)
            n_selected_tracks = len(selected_tracks)

            (
                n_matched_tracks,
                n_trackable_particles,
                n_matched_tracks_ROI,
                n_trackable_particles_ROI,
            ) = evaluate_tracks(
                truth,
                particles,
                tracks,
                frac_reco_matched=0.5,
                frac_truth_matched=0.5,
                ROI=ROI,
            )

            # IGNORE ROI (Region of Interest) FOR NOW...

            efficiency_matrix[i, j] = n_matched_tracks / n_trackable_particles
            fake_rate_matrix[i, j] = n_matched_tracks / n_selected_tracks

    return efficiency_matrix, fake_rate_matrix, edge_cut_range, track_length_range


def obtain_target_fake_rate(
    efficiency_matrix,
    fake_rate_matrix,
    edge_cut_range,
    track_length_range,
    target_fake_rate=1e-3,
):

    edge_cut_matrix = np.tile(edge_cut_range, (len(track_length_range), 1)).T
    track_length_matrix = np.tile(track_length_range, (len_edge_cut_range), 1)

    assert (
        fake_rate_matrix < target_fake_rate
    ).sum() > 0, "Cannot find any tracking values passing target fake rate"

    fake_rates_passing_cut = fake_rate_matrix < target_fake_rate
    best_fake_rate_arg = np.argmin(
        np.abs(fake_rate_matrix[fake_rates_passing_cut] - target_fake_rate)
    )

    best_fake_rate = fake_rate_matrix[fake_rates_passing_cut][best_fake_rate_arg]
    best_efficiency = edge_cut_matrix[fake_rates_passing_cut][best_fake_rate_arg]
    best_edge_cut = edge_cut_matrix[fake_rates_passing_cut][best_fake_rate_arg]
    best_track_length = track_length_matrix[fake_rates_passing_cut][best_fake_rate_arg]

    return best_fake_rate, best_efficiency, best_edge_cut, best_track_length


def evaluate_reco_tracks(
    truth: pd.DataFrame,
    reconstructed: pd.DataFrame,
    particles: pd.DataFrame,
    min_hits_truth=9,
    min_hits_reco=5,
    min_pt=1.0,
    frac_reco_matched=0.5,
    frac_truth_matched=0.5,
    **kwargs
):
    """Return


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
    # just in case particle_id == 0 included in truth.
    if "particle_id" in truth.columns:
        truth = truth[truth.particle_id > 0]

    # get number of spacepoints in each reconstructed tracks
    n_reco_hits = (
        reconstructed.track_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "track_id", "track_id": "n_reco_hits"})
    )

    # only tracks with a minimum number of spacepoints are considered
    n_reco_hits = n_reco_hits[n_reco_hits.n_reco_hits >= min_hits_reco]
    reconstructed = reconstructed[
        reconstructed.track_id.isin(n_reco_hits.track_id.values)
    ]

    # get number of spacepoints in each particle
    hits = truth.merge(particles, on="particle_id", how="left")
    n_true_hits = (
        hits.particle_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "particle_id", "particle_id": "n_true_hits"})
    )

    # only particles leaves at least min_hits_truth spacepoints
    # and with pT >= min_pt are considered.
    particles = particles.merge(n_true_hits, on=["particle_id"], how="left")

    is_trackable = particles.n_true_hits >= min_hits_truth

    # event has 3 columnes [track_id, particle_id, hit_id]
    event = pd.merge(reconstructed, truth, on=["hit_id"], how="left")

    # n_common_hits and n_shared should be exactly the same
    # for a specific track id and particle id

    # Each track_id will be assigned to multiple particles.
    # To determine which particle the track candidate is matched to,
    # we use the particle id that yields a maximum value of n_common_hits / n_reco_hits,
    # which means the majority of the spacepoints associated with the reconstructed
    # track candidate comes from that true track.
    # However, the other way may not be true.
    reco_matching = (
        event.groupby(["track_id", "particle_id"])
        .size()
        .reset_index()
        .rename(columns={0: "n_common_hits"})
    )

    # Each particle will be assigned to multiple reconstructed tracks
    truth_matching = (
        event.groupby(["particle_id", "track_id"])
        .size()
        .reset_index()
        .rename(columns={0: "n_shared"})
    )

    # add number of hits to each of the maching dataframe
    reco_matching = reco_matching.merge(n_reco_hits, on=["track_id"], how="left")
    truth_matching = truth_matching.merge(n_true_hits, on=["particle_id"], how="left")

    # calculate matching fraction
    reco_matching = reco_matching.assign(
        purity_reco=np.true_divide(
            reco_matching.n_common_hits, reco_matching.n_reco_hits
        )
    )
    truth_matching = truth_matching.assign(
        purity_true=np.true_divide(truth_matching.n_shared, truth_matching.n_true_hits)
    )

    # select the best match
    reco_matching["purity_reco_max"] = reco_matching.groupby("track_id")[
        "purity_reco"
    ].transform(max)
    truth_matching["purity_true_max"] = truth_matching.groupby("track_id")[
        "purity_true"
    ].transform(max)

    matched_reco_tracks = reco_matching[
        (reco_matching.purity_reco_max >= frac_reco_matched)
        & (reco_matching.purity_reco == reco_matching.purity_reco_max)
    ]

    matched_true_particles = truth_matching[
        (truth_matching.purity_true_max >= frac_truth_matched)
        & (truth_matching.purity_true == truth_matching.purity_true_max)
    ]

    # now, let's combine the two majority criteria
    # reconstructed tracks must be in both matched dataframe
    # and the so matched particle should be the same
    # in this way, each track should be only assigned
    combined_match = matched_true_particles.merge(
        matched_reco_tracks, on=["track_id", "particle_id"], how="inner"
    )

    n_reco_tracks = n_reco_hits.shape[0]
    n_true_tracks = particles.shape[0]

    # For GNN, there are non-negaliable cases where GNN-based
    # track candidates are matched to particles not considered as interesting.
    # which means there are paticles in matched_pids that do not exist in particles.
    matched_pids = np.unique(combined_match.particle_id)

    is_matched = particles.particle_id.isin(matched_pids).values
    n_matched_particles = np.sum(is_matched)

    n_matched_tracks = reco_matching[
        reco_matching.purity_reco >= frac_reco_matched
    ].shape[0]
    n_matched_tracks_poi = reco_matching[
        (reco_matching.purity_reco >= frac_reco_matched)
        & (reco_matching.particle_id.isin(particles.particle_id.values))
    ].shape[0]
    # print(n_matched_tracks_poi, n_matched_tracks)

    # num_particles_matched_to = reco_matched.groupby("particle_id")['track_id']\
    #     .count().reset_index().rename(columns={"track_id": "n_tracks_matched"})
    # n_duplicated_tracks = num_particles_matched_to.shape[0]
    n_duplicated_tracks = n_matched_tracks_poi - n_matched_particles

    particles = particles.assign(is_matched=is_matched, is_trackable=is_trackable)

    return (
        n_true_tracks,
        n_reco_tracks,
        n_matched_particles,
        n_matched_tracks,
        n_duplicated_tracks,
        n_matched_tracks_poi,
        particles,
    )


def run_one_evt(evtid, csv_reader, recotrkx_reader, **kwargs):
    # print("Running {}".format(evtid))

    raw_data = csv_reader(evtid)
    truth = raw_data.spacepoints[["hit_id", "particle_id"]]
    particles = raw_data.particles

    submission = recotrkx_reader(evtid)
    results = evaluate_reco_tracks(truth, submission, particles, **kwargs)
    return results[:-1] + (results[-1].assign(evtid=evtid),)
