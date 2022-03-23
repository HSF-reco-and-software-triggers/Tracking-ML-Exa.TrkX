import os
from functools import partial

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import itertools


fontsize=16
minor_size=14
pt_min, pt_max = 1000, 6000
default_pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 20)
default_pt_configs = {
    'bins': default_pt_bins,
    'histtype': 'step',
    'lw': 2,
    'log': False
}
default_eta_bins = np.arange(-4., 4.4, step=0.4)
default_eta_configs = {
    'bins': default_eta_bins,
    'histtype': 'step',
    'lw': 2,
    'log': False
}


def get_plot(nrows=1, ncols=1, figsize=6, nominor=False):

    fig, axs = plt.subplots(nrows, ncols,
        figsize=(figsize*ncols, figsize*nrows),
        constrained_layout=True)

    def format(ax):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        return ax

    if nrows * ncols == 1:
        ax = axs
        if not nominor: format(ax)
    else:
        ax = [format(x) if not nominor else x for x in axs.flatten()]

    return fig, ax

def add_up_xaxis(ax):
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(["" for x in ax.get_xticks()])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())

def get_ratio(x_vals, y_vals):
    res = [x/y if y!=0 else 0.0 for x,y in zip(x_vals, y_vals)]
    err = [x/y * math.sqrt((x+y)/(x*y)) if y!=0 and x!=0 else 0.0 for x,y in zip(x_vals, y_vals)]
    return res, err


def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)

def add_mean_std(array, x, y, ax, color='k', dy=0.3, digits=2, fontsize=12, with_std=True):
    this_mean, this_std = np.mean(array), np.std(array)
    ax.text(x, y, "Mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=12)
    if with_std:
        ax.text(x, y-dy, "Standard Deviation: {0:.{1}f}".format(this_std, digits), color=color, fontsize=12)

def make_cmp_plot(
    arrays, legends, configs,
    xlabel, ylabel, ratio_label,
    ratio_legends, ymin=0):

    _, ax = get_plot()
    vals_list = []
    for array,legend in zip(arrays, legends):
        vals, bins, _ = ax.hist(array, **configs, label=legend)
        vals_list.append(vals)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    add_up_xaxis(ax)
    ax.legend()
    ax.grid(True)
    plt.show()

    # make a ratio plot
    _, ax = get_plot()
    xvals = [0.5*(x[1]+x[0]) for x in pairwise(bins)]
    xerrs = [0.5*(x[1]-x[0]) for x in pairwise(bins)]

    for idx in range(1, len(arrays)):
        ratio, ratio_err = get_ratio(vals_list[-1], vals_list[idx-1])
        label = None if ratio_legends is None else ratio_legends[idx-1]
        ax.errorbar(
            xvals, ratio, yerr=ratio_err, fmt='o',
            xerr=xerrs, lw=2, label=label)


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ratio_label)
    add_up_xaxis(ax)

    if ratio_legends is not None:
        ax.legend()
    ax.grid(True)
    plt.show()


def plot_observable_performance(particles: pd.DataFrame):

    pt = particles.pt.values
    eta = particles.eta.values

    fiducial = (particles.status == 1) & (particles.barcode < 200000) & (particles.eta.abs() < 4) & (particles.radius < 260) & (particles.charge.abs() > 0) 
    trackable = particles.is_trackable
    matched = particles.is_double_matched


    # plot the performance `metric` as a function of `observable`
    make_cmp_plot_fn = partial(make_cmp_plot,
        legends=["Generated", "Reconstructable", "Matched"],
        ylabel="Num. particles", ratio_label='Track efficiency',
        ratio_legends=["Physics Eff", "Technical Eff"])

    all_cuts = [(1000, 4)]
    for (cut_pt, cut_eta) in all_cuts:
        cuts = (pt > cut_pt) & (np.abs(eta) < cut_eta)
        gen_pt = pt[cuts & fiducial]
        true_pt = pt[cuts & fiducial & trackable]
        reco_pt = pt[cuts & fiducial & trackable & matched]
        make_cmp_plot_fn([gen_pt, true_pt, reco_pt], 
            configs=default_pt_configs, xlabel="pT [MeV]", ymin=0.6)

        gen_eta = eta[cuts & fiducial]
        true_eta = eta[cuts & fiducial & trackable]
        reco_eta = eta[cuts & fiducial & trackable & matched]
        make_cmp_plot_fn([gen_eta, true_eta, reco_eta], configs=default_eta_configs, xlabel=r"$\eta$", ymin=0.6)
