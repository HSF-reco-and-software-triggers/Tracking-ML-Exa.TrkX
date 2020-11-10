import os, sys
import yaml
import importlib
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def find_config(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def find_checkpoint(run_id, path):
    for (root_dir, dirs, files) in os.walk(path):
        if run_id in dirs:
            latest_run_path = os.path.join(root_dir, run_id, "last.ckpt")
            return latest_run_path

def load_config(stage, resume_id, pipeline_config):
    if resume_id is None:
        with open(find_config(stage["config"], os.path.join(pipeline_config["model_library"], stage["set"]))) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    else:
        ckpnt_path = find_checkpoint(resume_id, pipeline_config["artifact_library"])
        ckpnt = torch.load(ckpnt_path, map_location=torch.device('cpu'))
        config = ckpnt["hyper_parameters"]

    if "override" in stage.keys():
        config.update(stage["override"])

    config["stage"] = stage

    return config

def find_model(model_set, model_name, model_library):

    # List all modules in the set/Models directory
    module_list = [os.path.splitext(name)[0] for
                   name in os.listdir(os.path.join(model_library, model_set, "Models")) if name.endswith(".py")]

    # Import all modules in the set/Models directory and find model_name
    imported_module_list = [importlib.import_module('.'.join([model_set, "Models", module])) for module in module_list]
    names = [mod for mod in imported_module_list if model_name in getattr(mod, '__all__', [n for n in dir(mod) if not n.startswith('_')])]

    # Return the class of model_name
    model_class = getattr(names[0], model_name)
    return model_class

def build_model(stage, pipeline_config):

    model_set = stage["set"]
    model_name = stage["name"]
    model_library = pipeline_config["model_library"]
    config_file = stage["config"]

    model_class = find_model(model_set, model_name, model_library)

    return model_class

def get_logger(model_config, run_id):

    wandb_logger = WandbLogger(project=model_config["project"], save_dir = model_config["wandb_save_dir"], id=run_id)

    return wandb_logger


def callback_objects(model_config):

    callback_list = model_config["callbacks"]
    model_set = model_config["stage"]["set"]
    callback_object_list = [find_model(model_set, callback)() for callback in callback_list]

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callback_object_list = callback_object_list + [lr_monitor]

    return

def build_trainer(model_config, logger, resume_id, pipeline_config):

    model_filepath = os.path.join(pipeline_config["artifact_library"], model_config["project"], logger.experiment._run_id, "last.ckpt")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filepath=model_filepath,
        save_top_k=1,
        save_last=True,
        mode='min')

    # Handle resume condition
    if resume_id is None:
        # The simple case: We start fresh
        trainer = pl.Trainer(max_epochs = model_config['max_epochs'], gpus=1, logger=logger, checkpoint_callback=checkpoint_callback, callbacks=callback_objects(model_config))
    else:
        # Here we assume
        trainer = pl.Trainer(resume_from_checkpoint=model_filepath, max_epochs = model_config['max_epochs'], gpus=1, logger=logger, checkpoint_callback=checkpoint_callback, callbacks=callback_objects(model_config))

    return trainer


def get_resume_id(stage):
    resume_id = stage["resume_id"] if "resume_id" in stage.keys() else None
    return resume_id
