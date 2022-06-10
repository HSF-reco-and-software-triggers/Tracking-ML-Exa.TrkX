# Track Matching Definitions

## Introduction

Given that each hit has been assigned (at least) one track label, we would like to quantify how well the labelling was performed. A key set of metrics are:
- What proportion of particles were "reconstructed" by a track
- What proportion of tracks were used to reconstruct particles, which we shall call being "matched" to a particle
Exactly how we define the goodness of reconstruction and matching is somewhat arbitrary, but there exist some common definitions.

## Styles of Matching

The following infographic explains the three styles of track matching used in this repository (click to zoom in).

[![](https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/matching_diagram.png)](https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/matching_diagram.png)
The three styles of matching - corresponding to `matching_style=` `ATLAS`, `one_way` and `two_way`