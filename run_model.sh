#!/bin/bash

MODEL="llama-2-7b-chat"
PY="generate_fine_tuning_dataset.py"

srun --gres=gpu:a100 -c 32 --mem=64G -t 10 torchrun --nproc_per_node 1 $PY --ckpt_dir $MODEL --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 20 $*
