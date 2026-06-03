#!/bin/bash

set -e

cd /root/chaokaimind

echo "===== $(date) Start Training ====="

python -u trainer/train_pretrain.py \
  --data_path /root/chaokaimind/dataset/pretrain_t2t_mini.jsonl \
  --tokenizer_path /root/chaokaimind/model \
  --save_dir /root/autodl-tmp/out \
  --save_weight pretrain \
  --from_resume 0 \
  --log_interval 20 \
  --save_interval 500 \
  2>&1 | tee /root/autodl-tmp/train.log

echo "===== $(date) Training Finished ====="

sync

shutdown -h now