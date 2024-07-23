mkdir -p $SLURM_TMPDIR/mne_data
tar -axf $HOME/projects/def-ravanelm/datasets-open/$1.tar -C $SLURM_TMPDIR/mne_data
