import os, sys
import yaml
import importlib
import logging
from itertools import product
from more_itertools import collapse
import subprocess

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from simple_slurm import Slurm


def find_checkpoint(run_id, path):  # M
    for (root_dir, dirs, files) in os.walk(path):
        if run_id in dirs:
            latest_run_path = os.path.join(root_dir, run_id, "last.ckpt")
            return latest_run_path


def handle_config_cases(some_config):  # C

    """
    Simply used to standardise the possible config entries. We always want a list
    """
    if type(some_config) is list:
        return some_config
    if some_config is None:
        return []
    else:
        return [some_config]


def submit_batch(batch_config_file, config, project_config, running_id=None):  # C

    with open(config["batch"]) as f:
        batch_config = yaml.load(f, Loader=yaml.FullLoader)

    command_line_args = dict_to_args(config)
    slurm = Slurm(**batch_config)

    if running_id is not None:
        print("Dependency on ID: {}".format(running_id))
        slurm.set_dependency(dict(afterok=running_id))

    custom_batch_setup = project_config["custom_batch_setup"]

    # Run the slurm submission command
    slurm_command = (
        "\n".join(custom_batch_setup) + """\n exabatch """ + command_line_args
    )

    # If there are setup commands to run before the batch submission, prepend them here
    if (
        "batch_setup" in config
        and config["batch_setup"]
        and len(project_config["command_line_setup"]) > 0
    ):
        slurm_setup = ";".join(project_config["command_line_setup"] + ["sbatch"])
    else:
        slurm_setup = "sbatch"

    logging.info(slurm_command)
    job_id = slurm.sbatch(slurm_command, sbatch_cmd=slurm_setup, shell="/bin/bash")
    print("Job ID: {}".format(job_id))
    return job_id


def find_config(name, path):  # C
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def load_config(stage, resume_id, project_config):  # C

    if resume_id is None:

        stage_config_file = find_config(
            stage["config"],
            os.path.join(project_config["libraries"]["model_library"], stage["set"]),
        )

        print(stage_config_file)

        with open(stage_config_file) as f:
            config = yaml.load(os.path.expandvars(f.read()), Loader=yaml.FullLoader)
        config["logger"] = project_config["logger"]
        config["resume_id"] = resume_id

    else:
        ckpnt_path = find_checkpoint(
            resume_id, project_config["libraries"]["artifact_library"]
        )
        ckpnt = torch.load(ckpnt_path, map_location=torch.device("cpu"))
        config = ckpnt["hyper_parameters"]

    if "override" in stage.keys():
        config.update(stage["override"])

    # Add pipeline configs to model_config
    config.update(project_config["libraries"])
    config.update(stage)

    logging.info("Config found and built")
    return config


def combo_config(config):  # C
    total_list = {k: (v if type(v) == list else [v]) for (k, v) in config.items()}
    keys, values = zip(*total_list.items())

    # Build list of config dictionaries
    config_list = []
    [config_list.append(dict(zip(keys, bundle))) for bundle in product(*values)]

    return config_list


def dict_to_args(config):  # C
    collapsed_list = list(collapse([["--" + k, v] for k, v in config.items()]))
    collapsed_list = [str(entry) for entry in collapsed_list]
    command_line_args = " ".join(collapsed_list)

    return command_line_args
