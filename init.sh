module load python 
conda activate exatrkx_hsf
if [ $NERSC_HOST = cori ]; then
    module load cgpu
fi