#!/bin/bash

set -e
set -o pipefail

cd /root/chaokaimind

export OMP_NUM_THREADS=4

echo "===== $(date) Start Full SFT ====="

python -u trainer/train_full_sft.py \
  --data_path /root/chaokaimind/dataset/sft_t2t_mini.jsonl \
  --tokenizer_path /root/chaokaimind/model \
  --save_dir /root/autodl-tmp/out \
  --save_weight full_sft \
  --from_weight pretrain \
  --from_resume 0 \
  --epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --dtype bfloat16 \
  --accumulation_steps 1 \
  --log_interval 20 \
  --save_interval 500 \
  2>&1 | tee /root/autodl-tmp/sft.log

echo "===== $(date) Full SFT Finished ====="

sync

shutdown -h now