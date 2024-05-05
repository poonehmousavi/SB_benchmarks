#!/bin/bash

python train.py hparams/MotorImagery/BNCI2014001/ssl.yaml \
    --data_folder '/data/BNCI2014001' \
    --cached_data_folder '/data' \
    --output_folder '/results/single-fold-example-braindecode/BNCI2014001' \
    --data_iterator_name 'leave-one-session-out' \
    --target_subject_idx 0 \
    --target_session_idx 1 \
    --number_of_epochs 50 \
    --device 'cuda'