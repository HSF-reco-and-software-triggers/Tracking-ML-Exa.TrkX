module load python 
conda activate extrkx_hsf_clone
if [ $NERSC_HOST = cori ]; then
    module load cgpu
fi

export EXATRKX_WD=$(pwd)