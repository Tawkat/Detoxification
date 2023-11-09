#!/bin/bash
#You can submit as interactive jobs for 24 hours or submit as a job for 28 days please select one of them


echo "Submitting job"
sbatch --time=0-11:30:00 --ntasks=1 --mem=256G --gres=gpu:4 --cpus-per-task=4 --account=rrg-mageed --output=./output_print/output_ft_no_expl_prompt_llama_7b_chat_bs8_2 --mail-user=tawkat97@gmail.com --mail-type=ALL --job-name=NPLC_bs8_2 no_expl_mgpu_train-jasmine.sh


