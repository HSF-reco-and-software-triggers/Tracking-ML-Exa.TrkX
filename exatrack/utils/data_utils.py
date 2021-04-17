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


def boolify(s):  # D
    if s == "True" or s == "true":
        return True
    if s == "False" or s == "false":
        return False
    raise ValueError("Not Boolean Value!")


def nullify(s):  # D
    if s == "None" or s == "none":
        return None
    raise ValueError("Not None type!")


def estimateType(var):  # D
    """guesses the str representation of the variables type"""
    if type(var) is list:
        if len(var) == 1:
            return estimateType(var[0])
        else:
            return [estimateType(varEntry) for varEntry in var]
    else:
        var = str(var)  # important if the parameters aren't strings...
        for caster in (nullify, boolify, int, float):
            try:
                return caster(var)
            except ValueError:
                pass
    return var


def autocast(dFxn):  # D
    def wrapped(*c, **d):
        cp = [estimateType(x) for x in c]
        dp = dict((i, estimateType(j)) for (i, j) in d.items())
        return dFxn(*cp, **dp)

    return wrapped
