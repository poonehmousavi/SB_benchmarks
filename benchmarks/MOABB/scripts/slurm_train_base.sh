source $HOME/speechbrain-benchmarks-private/benchmarks/MOABB/scripts/slurm_install_env.sh
$HOME/speechbrain-benchmarks-private/benchmarks/MOABB/scripts/slurm_cp_dataset.sh $dataset_tag

cd $HOME/speechbrain-benchmarks-private/benchmarks/MOABB
python train.py $@ --data_folder=$SLURM_TMPDIR/mne_data --cached_data_folder=$SLURM_TMPDIR/mne_data/pkl
