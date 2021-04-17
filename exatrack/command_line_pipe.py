"""The entry-point for the command line script `exapipe`.

This module parses the command-line arguments given to the entry-point of the Exatrack pipeline. 

Example:
    One would run this simply with `exapipe`, which will load in a `config/pipeline_test.yaml` pipeline config file. Provided that file exists and is well-defined (see docs or examples) then the pipeline will start to run. To test behaviour on a Slurm system, run `exapipe --batch`. 
    
Attributes:
   - pipeline_config: The yaml file listing the stages to run, in serial. If in batch mode, this will generate Slurm dependencies to ensure they run in series (NB: This only works if running on the same cluster! E.g. Cori GPU and CPU can NOT see each others dependencies at this time)
   - batch: Whether or not to run the pipeline stages in Slurm. If so, it will submit each stage separately, with the options defined in the pipeline file. That is, each stage can define a `batch` option that links to a file with the batch configuration for that stage (see docs for examples).
   - verbose: Self-explanatory
   
Todo:
  * Remove unnecessary argparse options, e.g. run-stage and batch_config. These are outdated from a previous implementation
  * Also, therefore, consolidate the outdated `run_parser` and `model_parser` into a single parser
  * Comment the `parse_pipeline` function more clearly
"""


import sys, os
import argparse
import yaml
import logging

from exatrack import run_pipeline


def parse_pipeline():

    """Parse command line arguments."""

    run_parser, model_parser = (
        argparse.ArgumentParser("run_pipeline.py"),
        argparse.ArgumentParser("run_pipeline.py"),
    )
    add_run_arg, add_model_arg = run_parser.add_argument, model_parser.add_argument
    add_run_arg("--batch", action="store_true")
    add_run_arg("--verbose", action="store_true")
    add_run_arg("--run-stage", action="store_true")
    add_run_arg("pipeline_config", nargs="?", default="configs/pipeline_test.yaml")
    add_run_arg("batch_config", nargs="?", default="configs/batch_gpu_default.yaml")

    run_parsed, model_to_parse = run_parser.parse_known_args()
    [
        add_model_arg(arg, nargs="+")
        for arg in model_to_parse
        if arg.startswith(("-", "--"))
    ]

    run_parsed, _ = run_parser.parse_known_args()
    model_parsed, _ = model_parser.parse_known_args()

    return run_parsed, model_parsed


def main():

    print("Running from top with args:", sys.argv)
    run_args, model_args = parse_pipeline()

    logging_level = logging.INFO if run_args.verbose else logging.WARNING
    logging.basicConfig(level=logging_level)
    #     logging.basicConfig(level=logging.INFO)
    logging.info("Parsed run args: {}".format(run_args))
    logging.info("Parsed model args: {}".format(model_args))

    if run_args.run_stage:
        run_pipeline.run_stage(**vars(model_args))
    else:
        run_pipeline.start(run_args)


if __name__ == "__main__":
    main()
