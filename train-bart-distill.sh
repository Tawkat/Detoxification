#!/bin/bash

module load gcc cuda python/3.8 ffmpeg/4.3.2 arrow/9.0.0 git-lfs
source ~/ENV_1/bin/activate



python3 ./train-t5-distill.py \
    --model_name_or_path "facebook/bart-base" \
    --do_train \
    --do_eval \
    --do_predict \
    --source_prefix "" \
    --train_file 'Datasets/Toxicity_Specific_Platform_3K/train/train_Specific_Platform_3K.csv' \
    --validation_file 'Datasets/Toxicity_Specific_Platform_3K/validation/val_Specific_Platform_3K.csv' \
    --test_file 'Datasets/Toxicity_All_Platform_3K/test/test_All_Platform_3K.csv' \
    --source_column "toxic" \
    --target_column "non_toxic" \
    --explanation_column "explanation" \
    --output_dir "Output_Dir/Training_Distill_Bart_1/Training_Platform_1/" \
    --overwrite_output_dir \
    --cache_dir ./cache \
    --seed 42 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_source_length 128 \
    --max_target_length 256 \
    --val_max_target_length 256\
    --gradient_checkpointing \
    --report_to="none" \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --predict_with_generate True \
    --metric_for_best_model "bleu" \
    --greater_is_better True \
    --load_best_model_at_end True \
    #--save_steps 1000 \
    #--eval_steps 1000 \
    #--logging_steps 1000 \
    #--deepspeed "ds_config2.json" \
    # --debugging \
    
deactivate
