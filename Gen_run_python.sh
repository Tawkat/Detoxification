#!/bin/bash

module load python/3.10
module load gcc/9.3.0 arrow
module load cuda/11.1.1 cudnn ffmpeg/4.3.2

source ~/ENV_1/bin/activate


echo "$4"
python3 ./Generate_Detox.py "$1" "$2" "$3" "$4"
