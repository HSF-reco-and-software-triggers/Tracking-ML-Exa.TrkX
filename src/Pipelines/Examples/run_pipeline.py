import sys, os
import argparse
import yaml
import logging

from simple_slurm import Slurm

from utils.stage_utils import get_resume_id, load_config, combo_config, dict_to_args, get_logger, build_model, build_trainer, autocast


def parse_pipeline():

    """Parse command line arguments."""

    parser = argparse.ArgumentParser('run_pipeline.py')
    add_arg = parser.add_argument
    add_arg('--batch', action="store_true")
    add_arg('--verbose', action="store_true")
    add_arg('--run-stage', action="store_true")
    add_arg('pipeline_config', nargs='?', default='configs/pipeline_default.yaml')
    add_arg('batch_config', nargs='?', default='configs/batch_default.yaml')

    parsed, unknown = parser.parse_known_args()
    
    [add_arg(arg, nargs="+") for arg in unknown if arg.startswith(("-", "--"))]
    
    return parser.parse_args()

def main(args):

    with open(args.pipeline_config) as f:
        pipeline_config = yaml.load(f, Loader=yaml.FullLoader)
    
    libraries = pipeline_config["libraries"]
    
    with open(args.batch_config) as f:
        batch_config = yaml.load(f, Loader=yaml.FullLoader)

    # Make models available to the pipeline
    sys.path.append(libraries["model_library"]) #  !!  TEST WITHOUT THIS LINE IN MAIN()

    for stage in pipeline_config["stage_list"]:

        # Set resume_id if it is given, else it is None and new model is built
        resume_id = get_resume_id(stage)

        # Get config file, from given location OR from ckpnt
        model_config = load_config(stage, resume_id, libraries)        
        model_config_combos = combo_config(model_config) if resume_id is None else [model_config]
    
        for config in model_config_combos:
            if args.batch:
                command_line_args = dict_to_args(config)
                slurm = Slurm(**batch_config)
                slurm.sbatch("""bash
                             conda activate exatrkx-test
                             python run_pipeline.py --run-stage """ + command_line_args)
            else:
                run_stage(**config)
                

@autocast        
def run_stage(**model_config):
    
    print("Running stage, with args")
    sys.path.append(model_config["model_library"])
    
    # Load the model and configuration file for this stage
    model_class = build_model(model_config)
    model = model_class(model_config)

    # Test if the model is TRAINABLE (i.e. a learning stage) or NONTRAINABLE (i.e. a processing stage)
    if callable(getattr(model, "training_step", None)):
        train_stage(model, model_config)
    else:
        data_stage(model, model_config)
    
    
def train_stage(model, model_config):
    
    # Define a logger (default: Weights & Biases)
    logger = get_logger(model_config)
    
    # Load the trainer, handling any resume conditions
    trainer = build_trainer(model_config, logger)

    # Run training
    trainer.fit(model)

    # Run testing and, if requested, inference callbacks to continue the pipeline
    trainer.test()

    
def data_stage(model, model_config):
    
    model.prepare_data()
    
if __name__=="__main__":

    print("Running from top with args:", sys.argv)
    args = parse_pipeline()
    
    logging_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=logging_level)
    logging.info("Parsed args:", args)
    
    if args.run_stage:
        run_stage(**vars(args))
    else:
        main(args)
