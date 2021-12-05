#!/bin/bash
module load StdEnv/2020 gcc/9.3.0 cuda/11.0
module load arrow/5.0.0
module load python/3.8
source $HOME/envABERT/bin/activate

OUTPUT_DIR=/home/mcao610/scratch/BART_distillation

# --source_prefix "summarize: " \
# --dataset_config "3.0.0" \
accelerate launch run_summarization_no_trainer.py \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 8 \
    --model_name_or_path /home/mcao610/scratch/huggingface/bart-large \
    --dataset_name xsum \
    --output_dir $OUTPUT_DIR;