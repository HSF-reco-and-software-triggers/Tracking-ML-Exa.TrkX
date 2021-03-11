import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

dependencies = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        'sklearn',
        'pyyaml>=5.1',
        'pytorch-lightning',
        'decorator',
        'more_itertools',
        'simple_slurm',
        'memory_profiler',
        'trackml@ https://github.com/LAL/trackml-library/tarball/master#egg=trackml-3'
        ]     

setup(
    name="TrackingMLExatrkx",
    
    version='0.2.0',

    description="Models, pipelines, and utilities for solving tracking problems with machine learning.",

    author="Daniel Murnane",

    install_requires=dependencies,
    packages=find_packages(include=['src', 'src.*']),

    long_description=read('README.md'),
    
    license="Apache License, Version 2.0",
    keywords=["graph networks", "track finding", "tracking", "seeding", "GNN", "machine learning"],
    url="https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX"
)

