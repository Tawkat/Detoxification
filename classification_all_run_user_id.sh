#!/bin/bash
#You can submit as interactive jobs for 24 hours or submit as a job for 28 days please select one of them

list=("facebook" "reddit" "fox news" "twitter" "stormfront" "wikipedia" "HateCheck")


for task in "${list[@]}"
do
   echo "Submitting $task"
   sbatch --time=0-08:30:00 --ntasks=1 --mem=60G --gres=gpu:1 --cpus-per-task=4 --account=rrg-mageed --output=./output_print/train_"$task"_1 --mail-user=tawkat97@gmail.com --mail-type=ALL --job-name="$task" classification_all_run_python.sh "$task"

done

