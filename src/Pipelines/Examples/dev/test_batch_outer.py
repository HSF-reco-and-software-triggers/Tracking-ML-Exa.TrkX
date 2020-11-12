import argparse
import sys, os
import yaml

from simple_slurm import Slurm

# def parse_args():

#     """Parse command line arguments."""

#     parser = argparse.ArgumentParser('run_pipeline.py')
#     add_arg = parser.add_argument    
#     add_arg('batch_config', nargs='?', default='batch_default.yaml')
    
#     return parser.parse_args()
            

def main():
    
    with open('batch_default.yaml') as f:
        batch_config = yaml.load(f, Loader=yaml.FullLoader)

#     batch_config = {
#     "job_name": "test_slurm_library",
#     "qos": "debug",
#     "constraint": "haswell",    
#     "nodes": 1,
#     "time": 5,
#     "output": "logs/%x-%j.out",
#     "account": "m1759"
#     }
    slurm = Slurm(**batch_config)
#     slurm = Slurm(job_name="test_slurm_library", qos="debug", constraint="haswell", nodes=1, time=5, account="m1759", output="test_output.log")
    slurm.sbatch('python test_script.py')

#     os.system('python test_script.py --some command --line options')

if __name__=="__main__":

#     args = parse_args()

    main()