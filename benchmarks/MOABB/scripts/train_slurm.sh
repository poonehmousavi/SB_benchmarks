module load StdEnv/2020
module load python/3.10.2
module load scipy-stack

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index $SCRATCH/wheels/*
pip install --no-index h5py

cd $HOME/speechbrain-benchmarks-private/benchmarks/MOABB
pip install --no-index -r extra-requirements.txt
