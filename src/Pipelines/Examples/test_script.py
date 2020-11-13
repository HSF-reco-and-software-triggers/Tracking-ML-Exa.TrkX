import argparse
import sys

from utils.stage_utils import autocast

def parse_args():

    """Parse command line arguments."""

    parser = argparse.ArgumentParser('run_pipeline.py')
    add_arg = parser.add_argument
    parsed, unknown = parser.parse_known_args()
    
    [add_arg(arg, nargs="+") for arg in unknown if arg.startswith(("-", "--"))]
    
    return parser.parse_args()
            
@autocast
def main(**args):
    print("Running test, with args:")
    print(args)

if __name__=="__main__":

    args = parse_args()

    main(**vars(args))