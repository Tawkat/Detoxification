#!/bin/bash

module load gcc cuda python/3.8 ffmpeg/4.3.2 arrow/9.0.0 git-lfs
source ~/ENV_1/bin/activate

python3 ./ChatGPT_Gen_1.py

deactivate
