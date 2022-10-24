# Graphs for TrackML: Zero to Hero in 20 Minutes

**Goal**: Run a graph-based training and inference pipeline on TrackML events and validate the performance, in less than 20 minutes.

## Option A: Run in Colab Notebook

Absolutely the easiest way to get started. Visit the [TrackML Colab Quickstart](https://colab.research.google.com/github/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/master/Examples/TrackML_Quickstart/colab_quickstart.ipynb) notebook on Google Colab. The notebook will run the training and inference pipelines in a around 20 minutes.

## Option B: Run in Local Notebook

Assuming access to a GPU, you can run the [TrackML Quickstart](run_quickstart.ipynb) notebook. First, you will need to create a conda environment with
```
conda env create -f environment.yml python=3.9
conda activate trackml-quickstart
```
then use this environment as your notebook kernel.

## Option C: Run from Command Line

Assuming access to a GPU, you can run the same pipeline from the command line. To see the steps in detail, see the [TrackML Quickstart](run_quickstart.ipynb) notebook. 

### Step 1: Download the data

A small set of TrackML events have been pre-processed and stored in a NERSC public portal:
```
mkdir datasets
wget https://portal.nersc.gov/cfs/m3443/dtmurnane/TrackML_Example/trackml_quickstart_dataset.tar.gz -O datasets/trackml_quickstart_dataset.tar.gz
```

### Step 2: Run the scripts

Assuming the environment is installed correctly and the data is available, we can run the training and inference stages simply by running
```
python Scripts/Step_1_Train_Metric_Learning.py
```
then
```
python Scripts/Step_2_Run_Metric_Learning.py
```
and so on.

### Step 3: Validate the results

The final script will print out the track reconstruction efficiency and fake rate, given the matching style as described in `pipeline_config.yaml`. To dig further into these definitions, look through the [notebook](run_quickstart.ipynb) and [docs](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/performance/truth_definitions/).