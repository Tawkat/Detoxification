#!/bin/bash

module load gcc cuda python/3.8 ffmpeg/4.3.2 arrow/9.0.0 git-lfs
source ~/ENV_1/bin/activate

export TASK_NAME="$1"

python3 run_classification.py \
  --model_name_or_path "roberta-base" \
  --task_name $TASK_NAME \
  --train_file "Datasets/Platforms/$1_EN_combined_binary_train.csv" \
  --validation_file "Datasets/Platforms/$1_EN_combined_binary_validation.csv" \
  --test_file "Datasets/Platforms/$1_EN_combined_binary_validation.csv" \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 512 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --output_dir Output_Dir/$TASK_NAME \
  --overwrite_output_dir True \
  --seed 1512 \
  --load_best_model_at_end True \
  --greater_is_better True \
  --metric_for_best_model 'f1' \
  --save_strategy "epoch" \
  --evaluation_strategy "epoch"
deactivate
