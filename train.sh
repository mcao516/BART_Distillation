#!/bin/bash
module load StdEnv/2020 gcc/9.3.0 cuda/11.0
module load arrow/5.0.0
module load python/3.8
source $HOME/envABERT/bin/activate

MODEL_NAME_OR_PATH=$SCRATCH/huggingface/bart-large
OUTPUT_DIR=$SCRATCH/BART_distillation


accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name xsum \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 8 \
    --num_warmup_steps 500 \
    --learning_rate 5e-5 \
    --num_beams 6 \
    --overwrite_cache false \
    --output_dir $OUTPUT_DIR;
    # --source_prefix "summarize: " \
    # --dataset_config "3.0.0" \
    # --preprocessing_num_workers 20 \