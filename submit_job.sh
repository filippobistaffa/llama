#!/bin/bash

MODEL="llama-2-7b-chat"
PY="generate_fine_tuning_dataset.py"
OUTDIR="$LUSTRE/finetuning"
ID=$RANDOM

mkdir -p $OUTDIR

tmpfile=$(mktemp)
sbatch 1> $tmpfile <<EOF
#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --time=10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=$OUTDIR/$ID.err

module load pandas
srun torchrun --nproc_per_node 1 $PY --ckpt_dir $MODEL --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 20 --dataset_file $OUTDIR/$ID.csv $*
EOF
