import argparse
import yaml

def parse_args():
    
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser('run_pipeline.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/pipeline_default.yaml')
    
    return parser.parse_args()

def main(args):
    
    with open(args.pipeline_config) as f:
        pipeline_config = yaml.load(f, Loader=yaml.FullLoader)
    
    for stage in pipeline_config["model_list"]:

        # Set resume_id if it is given, else it is None and new model is built
        resume_id = get_resume_id(stage)

        # Get config file, from given location OR from ckpnt
        model_config = load_config(stage, resume_id)

        # Define a logger (default: Weights & Biases)
        logger = get_logger(model_config, resume_id)

        # Load the model and configuration file for this stage
        model_class = build_model(stage)
        model = model_class(model_config)

        # Load the trainer, handling any resume conditions
        trainer = build_trainer(model_config, logger, resume_id)

        # Run training
        trainer.fit(model)
        
        # Run testing and, if requested, inference callbacks to continue the pipeline
        trainer.test()
        
if __name__=="__main__":
    
    args = parse_args()
    
    main(args)