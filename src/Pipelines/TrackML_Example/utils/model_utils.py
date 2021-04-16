import os, sys
import yaml
import importlib
import logging
from itertools import product
from more_itertools import collapse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from simple_slurm import Slurm

from .config_utils import handle_config_cases


def find_model(model_set, model_name, model_library):  # M

    # List all modules in the set/ and set/Models directories
    module_list = [
        os.path.splitext(name)[0]
        for name in os.listdir(os.path.join(model_library, model_set, "Models"))
        if name.endswith(".py")
    ]

    # Import all modules in the set/Models directory and find model_name
    imported_module_list = [
        importlib.import_module(".".join([model_set, "Models", module]))
        for module in module_list
    ]
    names = [
        mod
        for mod in imported_module_list
        if model_name
        in getattr(mod, "__all__", [n for n in dir(mod) if not n.startswith("_")])
    ]
    # Return the class of model_name
    model_class = getattr(names[0], model_name)
    logging.info("Model found")
    return model_class


def build_model(model_config):  # M

    model_set = model_config["set"]
    model_name = model_config["name"]
    model_library = model_config["model_library"]
    config_file = model_config["config"]

    logging.info("Building model...")
    model_class = find_model(model_set, model_name, model_library)

    logging.info("Model built")
    return model_class


def get_logger(model_config):  # M

    logger_choice = model_config["logger"]

    if logger_choice == "wandb":
        logger = WandbLogger(
            project=model_config["project"],
            save_dir=model_config["artifact_library"],
            id=model_config["resume_id"],
        )

    elif logger_choice == "tb":
        logger = TensorBoardLogger(
            name=model_config["project"],
            save_dir=model_config["artifact_library"],
            version=model_config["resume_id"],
        )

    elif logger_choice == None:
        logger = None

    logging.info("Logger retrieved")
    return logger


def callback_objects(model_config, lr_logger=False):  # M

    callback_list = model_config["callbacks"]
    callback_list = handle_config_cases(callback_list)

    model_set = model_config["set"]
    model_library = model_config["model_library"]
    callback_object_list = [
        find_model(model_set, callback, model_library)() for callback in callback_list
    ]

    if lr_logger:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callback_object_list = callback_object_list + [lr_monitor]

    logging.info("Callbacks found")
    return callback_object_list


def build_trainer(model_config, logger):  # M

    #     model_filepath = os.path.join(model_config["artifact_library"], model_config["project"], logger.experiment._run_id, "last.ckpt")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=2, save_last=True, mode="min"
    )

    gpus = 1 if torch.cuda.is_available() else 0

    # Handle resume condition
    if model_config["resume_id"] is None:
        # The simple case: We start fresh
        trainer = pl.Trainer(
            max_epochs=model_config["max_epochs"],
            gpus=gpus,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=callback_objects(model_config),
        )
    else:
        # Here we assume
        trainer = pl.Trainer(
            resume_from_checkpoint=model_filepath,
            max_epochs=model_config["max_epochs"],
            gpus=gpus,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=callback_objects(model_config),
        )

    logging.info("Trainer built")
    return trainer


def get_resume_id(stage):  # M
    resume_id = stage["resume_id"] if "resume_id" in stage.keys() else None
    return resume_id
