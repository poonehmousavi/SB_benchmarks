module load StdEnv/2020
module load python/3.10.2
module load scipy-stack

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index "h5py<4.0.0" scikit-learn orion torch_geometric torchinfo
pip install --no-index $SCRATCH/wheels/EDFlib_Python-1.0.8-py3-none-any.whl
pip install --no-index $SCRATCH/wheels/edfio-0.4.3-py3-none-any.whl
pip install --no-index $SCRATCH/wheels/mne-1.7.1-py3-none-any.whl
pip install --no-index $SCRATCH/wheels/mne_bids-0.15.0-py3-none-any.whl
pip install --no-index $SCRATCH/wheels/pymatreader-0.0.32-py3-none-any.whl
pip install --no-index $SCRATCH/wheels/pyriemann-0.6-py2.py3-none-any.whl
pip install --no-index $SCRATCH/wheels/speechbrain-1.0.0-py3-none-any.whl
pip install --no-index --no-deps $SCRATCH/wheels/moabb-1.1.0-py3-none-any.whl


cd $HOME/speechbrain-benchmarks-private/benchmarks/MOABB
