#!/bin/sh
$HOME/speechbrain-benchmarks-private/benchmarks/MOABB/scripts/slurm_install_env.sh
module load StdEnv/2020
module load python/3.10.2
module load scipy-stack

source $SLURM_TMPDIR/venv/bin/activate
$HOME/speechbrain-benchmarks-private/benchmarks/MOABB/scripts/slurm_cp_dataset.sh $dataset_tag

cd $HOME/speechbrain-benchmarks-private/benchmarks/MOABB
./run_hparam_optimization.sh $@ \
                             --output_folder $SLURM_TMPDIR/output_$SLURM_JOBID \
                             --data_folder $SLURM_TMPDIR/mne_data/
tar -caf $output_folder/results.tar.gz -C $SLURM_TMPDIR $SLURM_TMPDIR/output_$SLURM_JOBID
