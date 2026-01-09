#!/usr/bin/env bash

bash PATH_TO_code_dir/SB_benchmarks/benchmarks/DASB/run_extraction.sh \
                --data_folder "PATH_TO_data_dir/LibriSpeech" \
                --output_folder "PATH_TO_out_dir/mimi/ASR_large/extraction/1000/flexicodec" \
                --tokenizer "flexicodec" \
                --dataset "PATH_TO_code_dir/SB_benchmarks/benchmarks/DASB/LibriSpeech" \
                --save_embedding True \
                --cached_data_folder "[PATH_TO_out_dir/flexicodecdasb_result//mimi/ASR_large/extraction/1000/" \
                --checkpoint_path [flexicodec_checkpoint_path] \