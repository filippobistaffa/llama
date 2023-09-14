#!/bin/bash
#SBATCH --job-name=llama-13b
#SBATCH --time=10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:2
#SBATCH --output=13b.out
#SBATCH --error=13b.err

module load pandas
srun torchrun --nproc_per_node 2 extract_skills.py --ckpt_dir llama-2-13b-chat/ --tokenizer_path tokenizer.model --max_seq_len 1024 --max_batch_size 1
