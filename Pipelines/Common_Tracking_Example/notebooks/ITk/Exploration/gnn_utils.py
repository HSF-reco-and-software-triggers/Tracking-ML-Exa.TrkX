from tkinter import Y
import torch
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy

from tqdm import tqdm

# Load model from checkpoint path
def load_model(checkpoint_path, model_type):

    if not checkpoint_path.endswith(".ckpt"):
        checkpoint_path = get_latest_checkpoint_file(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path)
    model = model_type.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)

    return model, checkpoint["hyper_parameters"]

def get_latest_checkpoint_file(checkpoint_dir):
    # Load all files
    checkpoint_files = os.listdir(checkpoint_dir)
    # Filter by .ckpt
    checkpoint_files = [os.path.join(checkpoint_dir, file) for file in checkpoint_files if file.endswith(".ckpt")]
    latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)

    return latest_checkpoint_file

def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))

# Inference function
def inference(model, dataset_type="train", num_events=100): 
    assert dataset_type in ["train", "val", "test"] , "dataset type must be one of ['train', 'val', 'test']"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get dataset
    dataset_name = f"{dataset_type}set"

    trainsplit_dict = {"train": 0, "val": 1, "test": 2}
    model.hparams["datatype_split"] = [1, 1, 1]
    model.hparams["datatype_split"][trainsplit_dict[dataset_type]] = num_events
    model.setup(stage="fit")
    dataset = getattr(model, dataset_name)

    # Loop over batches
    result_list = []
    for batch in tqdm(dataset):
        result_list.append(infer_event(model, batch.to(device)))

    return result_list

def infer_event(model, batch):
    """
    Run inference on a single event to get the useful event attributes and edge scores
    """
    with torch.no_grad():
        results = model.shared_evaluation(batch, 0 , log=False)
        if "score" in results.keys(): # Using GNN
            scores = results["score"][: int(len(results["score"]) / 2)]
        elif "preds" in results.keys(): # Using Filter
            scores = results["preds"]
        batch.scores = scores.cpu()
    
    return batch.cpu()

def get_topline_stats(results, edge_cut, signal_true_label="pid_signal", bkg_true_label="y_pid"):
    """
    Apply edge cut to results.scores and get overall AUC, efficiency and purity
    """

    total_pos, total_signal_true, total_bkg_true, total_signal_true_pos, total_bkg_true_pos = [0] * 5

    for result in tqdm(results):
        pos = result.scores > edge_cut
        signal_true = result[signal_true_label]
        bkg_true = result[bkg_true_label]
        signal_true_pos = signal_true & pos
        bkg_true_pos = bkg_true & pos

        total_pos += pos.sum()
        total_signal_true += signal_true.sum()
        total_bkg_true += bkg_true.sum()
        total_signal_true_pos += signal_true_pos.sum()
        total_bkg_true_pos += bkg_true_pos.sum()

        result.signal_true_pos = signal_true_pos
        result.bkg_true_pos = bkg_true_pos
        result.pos = pos
        result.signal_true = signal_true

    print("Signal efficiency:", total_signal_true_pos / total_signal_true, 
    "Signal purity:", total_signal_true_pos / total_pos,
    "Background purity:", total_bkg_true_pos / total_pos)

# Build edge eta list and pt list
def build_edge_eta_list(results):

    av_eta_pred, av_eta_signal_true, av_eta_signal_true_pos, av_eta_bkg_true_pos = [], [], [], []
    av_r_pred, av_r_signal_true, av_r_signal_true_pos, av_r_bkg_true_pos = [], [], [], []
    # av_pt_pred, av_pt_signal_true, av_pt_bkg_true = [], [], []

    for result in tqdm(results):
        edge_pos = result.edge_index[:, result.pos]
        edge_signal_true = result.edge_index[:, result.signal_true]
        edge_signal_true_pos = result.edge_index[:, result.signal_true_pos]
        edge_bkg_true_pos = result.edge_index[:, result.bkg_true_pos]

        eta_hits = calc_eta(result.x[:, 0], result.x[:, 2])
        av_eta_pred.append((eta_hits[edge_pos[0]] + eta_hits[edge_pos[1]]) / 2)
        av_eta_signal_true.append((eta_hits[edge_signal_true[0]] + eta_hits[edge_signal_true[1]]) / 2)
        av_eta_signal_true_pos.append((eta_hits[edge_signal_true_pos[0]] + eta_hits[edge_signal_true_pos[1]]) / 2)
        av_eta_bkg_true_pos.append((eta_hits[edge_bkg_true_pos[0]] + eta_hits[edge_bkg_true_pos[1]]) / 2)

        r_hits = result.x[:, 0]
        av_r_pred.append((r_hits[edge_pos[0]] + r_hits[edge_pos[1]]) / 2)
        av_r_signal_true.append((r_hits[edge_signal_true[0]] + r_hits[edge_signal_true[1]]) / 2)
        av_r_signal_true_pos.append((r_hits[edge_signal_true_pos[0]] + r_hits[edge_signal_true_pos[1]]) / 2)
        av_r_bkg_true_pos.append((r_hits[edge_bkg_true_pos[0]] + r_hits[edge_bkg_true_pos[1]]) / 2)


    # Concatenate all lists of tensors
    av_eta_pred = torch.cat(av_eta_pred)
    av_eta_signal_true = torch.cat(av_eta_signal_true)
    av_eta_signal_true_pos = torch.cat(av_eta_signal_true_pos)
    av_eta_bkg_true_pos = torch.cat(av_eta_bkg_true_pos)

    av_r_pred = torch.cat(av_r_pred)
    av_r_signal_true = torch.cat(av_r_signal_true)
    av_r_signal_true_pos = torch.cat(av_r_signal_true_pos)
    av_r_bkg_true_pos = torch.cat(av_r_bkg_true_pos)
    

    return av_eta_pred, av_eta_signal_true, av_eta_signal_true_pos, av_eta_bkg_true_pos, av_r_pred, av_r_signal_true, av_r_signal_true_pos, av_r_bkg_true_pos

# Build edge eta list and pt list
def build_edge_pt_list(results):
    
    av_pt_pred, av_pt_signal_true, av_signal_true_pos, av_bkg_true_pos = [], [], [], []

    for result in tqdm(results):
        edge_pos = result.edge_index[:, result.pos]
        edge_signal_true = result.edge_index[:, result.signal_true]
        edge_signal_true_pos = result.edge_index[:, result.signal_true_pos]
        edge_bkg_true_pos = result.edge_index[:, result.bkg_true_pos]

        pt_hits = result.pt
        av_pt_pred.append((pt_hits[edge_pos[0]] + pt_hits[edge_pos[1]]) / 2)
        av_pt_signal_true.append((pt_hits[edge_signal_true[0]] + pt_hits[edge_signal_true[1]]) / 2)
        av_signal_true_pos.append((pt_hits[edge_signal_true_pos[0]] + pt_hits[edge_signal_true_pos[1]]) / 2)
        av_bkg_true_pos.append((pt_hits[edge_bkg_true_pos[0]] + pt_hits[edge_bkg_true_pos[1]]) / 2)

    # Concatenate all lists of tensors
    av_pt_pred = torch.cat(av_pt_pred)
    av_pt_signal_true = torch.cat(av_pt_signal_true)
    av_signal_true_pos = torch.cat(av_signal_true_pos)
    av_bkg_true_pos = torch.cat(av_bkg_true_pos)

    return av_pt_pred, av_pt_signal_true, av_signal_true_pos, av_bkg_true_pos


def get_ratio(x, y):
    res = x/y
    res[res != res] = 0
    err = x/y * np.sqrt((x+y)/(x*y))
    err[err != err] = 0
    return res, err

def plot_metrics(av_pos, av_signal_true, av_signal_true_pos, av_bkg_true_pos, num_bins=10, bounds=(-4., 4.), common_axes=None, x_label="$\eta$", y_labels=["Efficiency", "Signal purity", "Background purity"], log_x=False):

    if log_x:
        bins = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), num_bins)
    else:
        bins = np.linspace(bounds[0], bounds[1], num_bins)

    signal_true_counts, _ = np.histogram(av_signal_true, bins=bins)
    signal_true_pos_counts, _ = np.histogram(av_signal_true_pos, bins=bins)
    pred_counts, _ = np.histogram(av_pos, bins=bins)
    bkg_true_pos_counts, _ = np.histogram(av_bkg_true_pos, bins=bins)

    eff, eff_err = get_ratio(signal_true_pos_counts, signal_true_counts)
    signal_purity, signal_purity_err = get_ratio(signal_true_pos_counts, pred_counts)
    bkg_purity, bkg_purity_err = get_ratio(bkg_true_pos_counts, pred_counts)

    centers = (bins[:-1] + bins[1:]) / 2

    # eff_ax, signal_pur_ax, bkg_pur_ax = None, None, None
    axes_list = [None, None, None]

    if common_axes is None:
        
        # Plot and return figures    
        for i in range(len(y_labels)):
            _, axes_list[i] = plt.subplots(1, 1, figsize=(10, 5))

    else:
        common_axes = common_axes if type(common_axes) in [list, np.ndarray] else [common_axes]
        axes_list = common_axes

    for ax, counts, err, label in zip(axes_list, [eff, signal_purity, bkg_purity], [eff_err, signal_purity_err, bkg_purity_err], y_labels):
        ax.errorbar(centers, counts, yerr=err, xerr=(bins[:-1] - bins[1:]) / 2, fmt="o", elinewidth=2, capsize=5, capthick=2)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)

        # Get color of the last plot
        color = ax.lines[-1].get_color()

        # Add a cubic spline to the efficiency plot to smooth it
        spline = scipy.interpolate.UnivariateSpline(centers, counts, s=0)
        spline_x = np.linspace(centers[0], centers[-1], 1000)
        spline_y = spline(spline_x)
        ax.plot(spline_x, spline_y, linestyle=":", color=color)

    return axes_list
    

def run_eta_performance(checkpoint_path, model_type, dataset_type, num_events, score_cut, common_axes = None, vmin=[None, None, None], vmax=[None, None, None], signal_true_label="pid_signal", bkg_true_label="y_pid"):
    model, _ = load_model(checkpoint_path, model_type)
    results = inference(model, dataset_type, num_events)
    get_topline_stats(results, score_cut, signal_true_label, bkg_true_label)
    av_eta_pred, av_eta_signal_true, av_eta_signal_true_pos, av_eta_bkg_true_pos, av_r_pred, av_r_signal_true, av_r_signal_true_pos, av_r_bkg_true_pos = build_edge_eta_list(results)
    eff_ax, signal_pur_ax, bkg_pur_ax = plot_metrics(av_eta_pred, av_eta_signal_true, av_eta_signal_true_pos, av_eta_bkg_true_pos, common_axes = common_axes)
    if common_axes is None:
        plot_eta_r_metrics(av_eta_pred, av_eta_signal_true, av_eta_signal_true_pos, av_eta_bkg_true_pos, av_r_pred, av_r_signal_true, av_r_signal_true_pos, av_r_bkg_true_pos, vmin=vmin, vmax=vmax)

    return eff_ax, signal_pur_ax, bkg_pur_ax


def run_pt_performance(checkpoint_path, model_type, dataset_type, num_events, score_cut, common_axes = None, signal_true_label="pid_signal", bkg_true_label="y_pid"):
    """
    Get performance of the model on the pt distribution
    """
    model, _ = load_model(checkpoint_path, model_type)
    results = inference(model, dataset_type, num_events)
    get_topline_stats(results, score_cut, signal_true_label, bkg_true_label)
    av_pt_pred, av_pt_signal_true, av_pt_signal_true_pos, av_pt_bkg_true_pos = build_edge_pt_list(results)
    eff_ax = plot_metrics(av_pt_pred, av_pt_signal_true, av_pt_signal_true_pos, av_pt_bkg_true_pos, num_bins=20, bounds=(1000, 100000), common_axes = common_axes, x_label="$p_{T} (MeV)$", y_labels=["Efficiency"], log_x=True)

    return eff_ax



    

def make_2d_ratio(counts_numerator, counts_denominator, eta_bins, r_bins, label="Efficiency", vmin=None, vmax=None):

    ratio = counts_numerator/ counts_denominator
    ratio[ratio != ratio] = 0.

    # Plot ratio as 2d histogram
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))

    if vmin is None:
        vmin = ratio[ratio != 0].min()*0.9
    if vmax is None:
        ratio[ratio != 0].max()
    im = ax.imshow(ratio.T, origin='lower', extent=[eta_bins[0], eta_bins[-1], r_bins[0], r_bins[-1]], vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$r$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    ax.grid(True)

def plot_eta_r_metrics(av_eta_pred, av_eta_signal_true, av_eta_signal_true_pos, av_eta_bkg_true_pos, av_r_pred, av_r_signal_true, av_r_signal_true_pos, av_r_bkg_true_pos, vmin=[None, None, None], vmax=[None, None, None]):

    default_eta_bins = np.arange(-4., 4.4, step=0.4)
    default_r_bins = np.arange(0., 1, step=0.1)

    # Make 2d histogram from eta and r values
    signal_true_2d_counts, _, _ = np.histogram2d(av_eta_signal_true.numpy(), av_r_signal_true.numpy(), bins=[default_eta_bins, default_r_bins])
    signal_true_pos_2d_counts, _, _ = np.histogram2d(av_eta_signal_true_pos.numpy(), av_r_signal_true_pos.numpy(), bins=[default_eta_bins, default_r_bins])
    pred_2d_counts, _, _ = np.histogram2d(av_eta_pred.numpy(), av_r_pred.numpy(), bins=[default_eta_bins, default_r_bins])
    bkg_true_pos_2d_counts, _, _ = np.histogram2d(av_eta_bkg_true_pos.numpy(), av_r_bkg_true_pos.numpy(), bins=[default_eta_bins, default_r_bins])

    make_2d_ratio(signal_true_pos_2d_counts, signal_true_2d_counts, default_eta_bins, default_r_bins, label="Efficiency", vmin=vmin[0], vmax=vmax[0])
    make_2d_ratio(signal_true_pos_2d_counts, pred_2d_counts, default_eta_bins, default_r_bins, label="Signal Purity", vmin=vmin[1], vmax=vmax[1])
    make_2d_ratio(bkg_true_pos_2d_counts, pred_2d_counts, default_eta_bins, default_r_bins, label="Background Purity", vmin=vmin[2], vmax=vmax[2])
