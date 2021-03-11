import sys, os
import argparse
import yaml
import logging

from utils.stage_utils import get_resume_id, load_config, combo_config, get_logger, build_model, build_trainer, autocast, submit_batch


def parse_pipeline():

    """Parse command line arguments."""

    run_parser, model_parser = argparse.ArgumentParser('run_pipeline.py'), argparse.ArgumentParser('run_pipeline.py')
    add_run_arg, add_model_arg = run_parser.add_argument, model_parser.add_argument
    add_run_arg('--batch', action="store_true")
    add_run_arg('--verbose', action="store_true")
    add_run_arg('--run-stage', action="store_true")
    add_run_arg('pipeline_config', nargs='?', default='configs/pipeline_default.yaml')
    add_run_arg('batch_config', nargs='?', default='configs/batch_gpu_default.yaml')

    run_parsed, model_to_parse = run_parser.parse_known_args()
    [add_model_arg(arg, nargs="+") for arg in model_to_parse if arg.startswith(("-", "--"))]
    
    run_parsed, _ = run_parser.parse_known_args()
    model_parsed, _ = model_parser.parse_known_args()
    
    return run_parsed, model_parsed               

@autocast        
def run_stage(**model_config):
    
    print("Running stage, with args, and model library:", model_config["model_library"])
    sys.path.append(model_config["model_library"])
    
    # Load the model and configuration file for this stage
    model_class = build_model(model_config)
    model = model_class(model_config)
    logging.info("Model found")

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
    logging.info("Preparing data")
    model.prepare_data()
    

def main(args):

    with open(args.pipeline_config) as f:
        pipeline_config = yaml.load(f, Loader=yaml.FullLoader)
    
    with open("configs/project_config.yaml") as f:
        project_config = yaml.load(f, Loader=yaml.FullLoader)
    
    libraries = project_config["libraries"]   

    # Make models available to the pipeline
    sys.path.append(libraries["model_library"]) #  !!  TEST WITHOUT THIS LINE IN MAIN()

    for stage in pipeline_config["stage_list"]:

        # Set resume_id if it is given, else it is None and new model is built
        resume_id = get_resume_id(stage)

        # Get config file, from given location OR from ckpnt
        model_config = load_config(stage, resume_id, libraries, project_config)   
        logging.info("Single config: {}".format(model_config))
        
        model_config_combos = combo_config(model_config) if resume_id is None else [model_config]
        logging.info("Combo configs: {}".format(model_config_combos))
        
        for config in model_config_combos:
            if args.batch:
                submit_batch(config, project_config)
            else:
                run_stage(**config)
    
    
if __name__=="__main__":

    print("Running from top with args:", sys.argv)
    run_args, model_args = parse_pipeline()
    
    logging_level = logging.INFO if run_args.verbose else logging.WARNING
#     logging.basicConfig(level=logging_level)
    logging.basicConfig(level=logging.INFO)
    logging.info("Parsed run args: {}".format(run_args))
    logging.info("Parsed model args: {}".format(model_args))
    
    if run_args.run_stage:
        run_stage(**vars(model_args))
    else:
        main(run_args)
