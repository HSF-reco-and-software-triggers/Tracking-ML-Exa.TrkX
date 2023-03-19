module load python 
conda activate py38torch113pyg22
if [ $NERSC_HOST = cori ]; then
    module load cgpu
fi

export EXATRKX_WD=$(pwd)