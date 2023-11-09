import os
import sys
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, pipeline
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import gc

from string import Template

DATASET = str(sys.argv[1]) #"Datasets/ParaDetox/ParaDetox_test.csv"

MODEL_DIR = str(sys.argv[2]) #"Output_Dir/No_Expl_LLama_Chat_7b_1/Llama-2-7b-chat-hf"

ckpt = str(sys.argv[3])

PROMPT = str(sys.argv[4]) #"Rewrite the following toxic input into non-toxic version. You must preserve the original meaning as much as possible.\n Input: "

print(PROMPT)

def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    #print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 2 and len(preds[i]) > 2:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1

    try:
        avg_bleu = float(bleu_sim / counter)
    except:
        avg_bleu = 0.0
    return avg_bleu

'''def process(text):
    #prompt = "Rewrite the following toxic input into non-toxic version. Let's break the input down step by step to rewrite the non-toxic version. You should first think about the expanation of why the input text is toxic. Then generate the detoxic output. You must preserve the original meaning as much as possible.\nInput: "
    text = PROMPT+text+"\n"
    return text
'''

def process(text):
    prompt_template = Template(PROMPT)
    new_text = prompt_template.substitute(toxic = text)
    return new_text


def get_detox(text):
    detox_list = text.split('Detoxification:')
    if(len(detox_list)==1):
        return ''
    detox = detox_list[-1].strip()
    #expl = detox_list[0].split('Explanation:')[-1].strip()
    return detox
  

print(f"{DATASET.split('/')[1]}")
df_test = pd.read_csv(DATASET)
df_test['input'] = df_test['toxic'].apply(process)

checkpoints = os.listdir(MODEL_DIR)

result_file = os.path.join(MODEL_DIR, f"results_{DATASET.split('/')[1]}.xlsx")
    
if not os.path.exists(result_file):
    print('*** CREATING new Excel file ... ***')

    df_result = {}
    df_result['Model_Dir'] = []
    df_result[f'BLEU'] = []
else:
    print('*** LOADING EXISTING Resulting file ***')
    df_result = pd.read_excel(result_file)
    df_result = df_result.to_dict(orient='list')




'''for ckpt in tqdm(checkpoints):
  if(not ckpt.startswith('checkpoint')):
    continue'''
  
path = os.path.join(MODEL_DIR, ckpt)
print(f"***** CHECKPOINT: {path}  *******")

tokenizer = LlamaTokenizer.from_pretrained(path)
model = LlamaForCausalLM.from_pretrained(path)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device = 0)
gen_list = generator(df_test['input'].tolist(), max_new_tokens=512) ### 512 

predictions = []
output_list = []
#pred_explanations = []

for i,gen in enumerate(gen_list):
    #print(gen[0])
    #try:
    pred = get_detox(gen[0]['generated_text'])
    predictions.append(pred)
    output_list.append(gen[0]['generated_text'].strip())
    #pred_explanations.append(expl)
    '''except:
        print(i)
        print(gen[0]['generated_text'])
        print("******")'''

sources = df_test['toxic'].tolist()
references = df_test['non_toxic'].tolist()
#ref_explanations = df_test['explanation'].tolist()


output_prediction_file = os.path.join(path, f"{DATASET.split('/')[1]}_generated_predictions.jsonl")

with open(output_prediction_file, "w") as writer:             
    for src, ref, pred, op in zip(sources, references, predictions, output_list):
        output = {
            'source': src,
            'reference': ref,
            'prediction': pred,
            'response': op,
            #'predicted_explanation': pred_expl,
            #'ChatGPT_explanation': ref_expl,
        }
        json.dump(output, writer)
        #writer.write("REFERENCES: "+str(ref))
        #writer.write("\nPREDICTION: "+str(pred))
        writer.write("\n")
df_predicted = pd.read_json(output_prediction_file, lines=True)

df_predicted['index'] = df_predicted.index

df_predicted.to_excel(f"{path}/{DATASET.split('/')[1]}_generated_predictions.xlsx", index=False)

bleu = calc_bleu(sources, predictions)

df_result['Model_Dir'].append(str(ckpt))
df_result['BLEU'].append(str(bleu))
#print(df_result)
df_result = pd.DataFrame(df_result)
print(f'*** Saving file at: {result_file} ***')
df_result.to_excel(result_file, index=False)

#del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()
  

    
print(f"***** BLEU: {bleu} *****")
#print(f"***** BEST CHECKPOINT: {best_ckpt} *****")

