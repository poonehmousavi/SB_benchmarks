source $HOME/speechbrain-benchmarks-private/benchmarks/MOABB/scripts/slurm_install_env.sh

mkdir -p $SLURM_TMPDIR/mne_data
tar -axf $HOME/projects/def-ravanelm/datasets-open/${dataset_tag}.tar -C $SLURM_TMPDIR/mne_data

cd $HOME/speechbrain-benchmarks-private/benchmarks/MOABB
python train.py $@ --data_folder=$SLURM_TMPDIR/mne_data --cached_data_folder=$SLURM_TMPDIR/mne_data/pkl
