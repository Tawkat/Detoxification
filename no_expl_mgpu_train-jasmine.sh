#!/bin/bash
/bin/hostname -s

module load python/3.10
module load gcc/9.3.0 arrow/9.0.0
module load cuda/11.1.1 cudnn ffmpeg/4.3.2
source ~/ENV_1/bin/activate

#export NCCL_BLOCKING_WAIT=1
#export WANDB_DISABLED=true
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

#echo "Num of node, $SLURM_JOB_NUM_NODES"
#echo "Num of GPU per node, $NPROC_PER_NODE"
#echo "PROCID: $SLURM_PROCID"
#echo "LOCALID: $SLURM_LOCALID"

#nvcc --version

#deepspeed --num_gpus=$NUM_GPUS --num_nodes=1 

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export HDF5_USE_FILE_LOCKING='FALSE'
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1



SLURM_GPUS_PER_NODE=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ $SLURM_JOB_NUM_NODES ] 
then
    echo "Num of node: $SLURM_JOB_NUM_NODES"
else
    echo "Num of node: 1"
    SLURM_JOB_NUM_NODES=1
fi

echo "Training started at $(date)"
echo "Num of node: $SLURM_JOB_NUM_NODES"
echo "Num of GPU per node: $SLURM_GPUS_PER_NODE"
echo "PROCID: $SLURM_PROCID"
echo "LOCALID: $SLURM_LOCALID"
# echo "model_name: $model_name"
# echo "Dataset: $dataset"
# echo "Epochs: $epochs"

echo "Training started at $(date)"


accelerate launch --config_file config_mim.yaml no_expl_train-jasmine.py \
    --model_name_or_path "models/Llama-2-7b-chat-hf" \
    --data_dir 'Datasets/' \
    --train_data_path 'Datasets/Toxicity_Specific_Platform_3K/train/train_Specific_Platform_3K.csv' \
    --eval_data_path 'Datasets/Toxicity_Specific_Platform_3K/validation/val_Specific_Platform_3K.csv' \
    --predict_data_path 'Datasets/Toxicity_All_Platform_3K/test/test_All_Platform_3K.csv' \
    --output_dir "Output_Dir/No_Expl_Prompt_LLama_Chat_7b_bs8_512_2" \
    --cache_dir ./cache \
    --seed 42 \
    --bf16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --source_max_length 512 \
    --target_max_length 512 \
    --gradient_checkpointing \
    --report_to="none" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1000 \
    --predict_with_generate True \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --load_best_model_at_end True  
    # --debugging \
