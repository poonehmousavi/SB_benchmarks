#!/usr/bin/env 

bash PATH_TO_code_dir/SB_benchmarks/benchmarks/DASB/run_experiments.sh \
  --hparams PATH_TO_code_dir/SB_benchmarks/benchmarks/DASB/LibriSpeech/ASR/hparams/LSTM/train.yaml \
  --data_folder "PATH_TO_data_dir/LibriSpeech" \
  --cached_data_folder "PATH_TO_cache_dir/mimi/ASR_large/extraction/1000/" \
  --output_folder "PATH_TO_out_dir/flexicodecdasb_result/mimi/ASR/flexicodec_${name}_correct/1cb/LSTM-pretrain-1024" \
  --task "ASR" \
  --dataset "LibriSpeech" \
  --seed 1986 \
  --nruns 1 \
  --eval_metric "WER" \
  --token_type "bpe" \
  --output_neurons "500" \
  --tokens_folder "PATH_TO_tokens_dir/ASR_large/extraction/1000/flexicodec_${name}/save/librispeech/" \
  --num_codebooks "1" \
  --sample_rate "44100" \
  --vocab_size "1024" \
  --batch_size "4" \
  --dnn_layers "3" \
  --lr_model "0.0002548" \
  --hidden_dim "128" \
  --encoder_dim "1024" \
  --pretrain_embeddings "True" \
  --pretrain_embeddings_folder "PATH_TO_tokens_dir/flexicodecdasb_result/mimi/ASR_large/extraction/1000/flexicodec_${name}/save/embeddings/" \
  --number_of_epochs 30 \