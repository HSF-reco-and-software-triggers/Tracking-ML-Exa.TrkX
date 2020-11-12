import sys, os
import argparse
import yaml
import logging

from simple_slurm import Slurm

from utils.stage_utils import get_resume_id, load_config, combo_config, dict_to_args, get_logger, build_model, build_trainer, autocast
logging.basicConfig(level=logging.WARNING)


def parse_pipeline():

    """Parse command line arguments."""

    parser = argparse.ArgumentParser('run_pipeline.py')
    add_arg = parser.add_argument
    add_arg('--batch', action="store_true")
    add_arg('--run-stage', action="store_true")
    add_arg('pipeline_config', nargs='?', default='configs/pipeline_default.yaml')
    add_arg('batch_config', nargs='?', default='configs/batch_default.yaml')

    parsed, unknown = parser.parse_known_args()
    
    [add_arg(arg, nargs="+") for arg in unknown if arg.startswith(("-", "--"))]
    
    return parser.parse_args()

def parse_stage():

    """Parse command line arguments."""

    parser = argparse.ArgumentParser('run_pipeline.py')
    add_arg = parser.add_argument
    parsed, unknown = parser.parse_known_args()
    
    [add_arg(arg, nargs="+") for arg in unknown if arg.startswith(("-", "--"))]
    
    return parser.parse_args()

def main(args):

    with open(args.pipeline_config) as f:
        pipeline_config = yaml.load(f, Loader=yaml.FullLoader)
    
    print("PIPELINE CONFIG:", pipeline_config)
    
    with open(args.batch_config) as f:
        batch_config = yaml.load(f, Loader=yaml.FullLoader)

    print("SLURM CONFIG:", batch_config)
    # Make models available to the pipeline
    sys.path.append(pipeline_config["model_library"])

    for stage in pipeline_config["stage_list"]:

        # Set resume_id if it is given, else it is None and new model is built
        resume_id = get_resume_id(stage)

        # Get config file, from given location OR from ckpnt
        model_config = load_config(stage, resume_id, pipeline_config)
#         print("Model configuration:", model_config)
        
        model_config_combos = combo_config(model_config)
#         print("Model configuration:", model_config_combos)
    
        print(model_config_combos[0])
    
        for config in model_config_combos:
            command_line_args = dict_to_args(config)
            print("ARGS TO COMMAND LINE:", command_line_args)
#             os.system('python test_script.py --run-stage ' + command_line_args)
            slurm = Slurm(**batch_config)
            slurm.sbatch('python test_script.py ' + command_line_args)

        
def run_stage(args):
    print("Running stage, with args:")
    print(args)
    
        # Define a logger (default: Weights & Biases)
#         logger = get_logger(model_config, resume_id)

#         # Load the model and configuration file for this stage
#         model_class = build_model(stage, pipeline_config)
#         model = model_class(model_config)

#         # Load the trainer, handling any resume conditions
#         trainer = build_trainer(model_config, logger, resume_id, pipeline_config)

#         # Run training
#         trainer.fit(model)

#         # Run testing and, if requested, inference callbacks to continue the pipeline
#         trainer.test()

if __name__=="__main__":

    print("Running from top with args:", sys.argv)
    args = parse_pipeline()
    print("Parsed args:", args)
    if args.run_stage:
        run_stage(args)
    else:
        main(args)
