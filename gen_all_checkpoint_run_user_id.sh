#!/bin/bash
#You can submit as interactive jobs for 24 hours or submit as a job for 28 days please select one of them

Dataset="Datasets/ParaDetox/ParaDetox_test.csv"
Model_Dir="Output_Dir/Prompt_LLama_Chat_7b_bs_8_1/Llama-2-7b-chat-hf/"
Prompt="[INST] <<SYS>>\nYou are a toxic to non-toxic style transfer (detoxification) model. Given a toxic input, you should first think and write about the expanation of why the input text is toxic. Then rewrite the toxic input into non-toxic version. The non-toxic output should not contain any form of toxic text. Also, You must preserve the original meaning as much as possible.\n<</SYS>>\n\n\$toxic [/INST] "

for model in  checkpoint-29 checkpoint-58 checkpoint-87 checkpoint-116 checkpoint-145 checkpoint-174 checkpoint-203 checkpoint-233 checkpoint-262 checkpoint-290
do
   echo "Submitting $model"
   sbatch --time=0-3:40:00 --ntasks=1 --mem=256G --gres=gpu:1 --cpus-per-task=4 --account=rrg-mageed --output=./output_print/Gen_PLC_bs8_512_"$model"_1 --mail-user=tawkat97@gmail.com --mail-type=ALL --job-name="$model" Gen_run_python.sh "$Dataset" "$Model_Dir" "$model" "$Prompt"

done

