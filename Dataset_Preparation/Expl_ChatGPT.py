
import pandas as pd
import numpy as np
import os
import openai
import time
import json

from string import Template
from tqdm import tqdm

from datetime import datetime
import concurrent.futures


SAMPLE_SIZE = 2000
EXTRA = 20

TOTAL_GEN = SAMPLE_SIZE # + EXTRA

PROMPTS = {
    'explanation': "You are a generative language model. Given an input text, your task is to explain thoroughly why it is toxic. This explanation will be used to finetune a Seq2Seq model to generate toxic language explanation.\nYou must meet the following requirement:\n1. Explanation should mention the type of toxicity. For example, you should whether the input text contain any offensive/curse word, any targeted  hate speech to a particular group, any personal attack, etc. You can include other relevant types that are not mentioned here.\n2. Try to provide a detailed and concise explanation why the given input text is toxic.\n3. Your explanation should not contain more than three sentences.\n\nNow provide the explanation why the following input text is toxic:\nInput: {{$input_text}}\nExplanation:",
}


#df_template = pd.read_excel('Datasets/text-style-transfer.xlsx', sheet_name='all_attribute')
#df_template.head()

df_data = {}
df_data['train'] = pd.read_csv('Datasets/Before_Exp_All_3K/train/train_All_Platform_3K.csv')
df_data['val'] = pd.read_csv('Datasets/Before_Exp_All_3K/validation/val_All_Platform_3K.csv')
df_data['test'] = pd.read_csv('Datasets/Before_Exp_All_3K/test/test_All_Platform_3K.csv')
#df_data.head()

out_cols = df_data['train'].columns


def process_response(response):
    response = response.replace('{','')
    response = response.replace('}','')
    response = response.split("Input:")[-1]
    response = response.split("input:")[-1]
    response = response.split("Text:")[-1]
    response = response.split("text:")[-1]
    response = response.split("Explanation:")[-1]
    response = response.split("explanation:")[-1]
    response = response.strip()
    return response


openai.api_key = open("OPENAI/openai_api_0.txt", "r").readline().strip()
model = "gpt-3.5-turbo"

def send_request(request):
    
    #print(request.items())
    
    output = {}
    
    chatgpt_keys = ['model', 'messages', 'max_tokens']
    
    for col in out_cols:
        #print(col)
        output[col] = str(request.pop(col))

    
    # add retry mechanism when face any issues, such as rate limit. Try 5 times and wait for 2 seconds each time
    for i in range(3):
        try:
            raw_responses = openai.ChatCompletion.create(**request)
            answer = raw_responses["choices"][0]["message"]["content"].replace("{", "").replace("}", "").strip()
            processed_answer = process_response(answer)

            output['explanation']=processed_answer
            #print(output.keys())
            # print(answer, gold_label)
            # print(answer, gold_label)
            with open(result_file, "a") as f:
                json.dump(output, f)
                f.write("\n")
            break
        except Exception as e:
            print(e)
            time.sleep(20)
            
            

            



global result_file
result_file = f"Generated_Data/Exp_All_Platform_3K/explanation_all.jsonl"
if os.path.exists(result_file):
    os.remove(result_file)


for split in df_data.keys():

    
    df_data_tmp = pd.DataFrame(df_data[split])
    #print(len(df_data_tmp['toxic']))

    all_requests = []

    for k in tqdm(range(len(df_data_tmp['toxic']))):
        
        prompt_template = Template(PROMPTS['explanation'])
        prompt_indiv = prompt_template.substitute(input_text = df_data_tmp['toxic'][k])
        #print(prompt_indiv)
        #prompt_indiv = prompt_indiv.replace('\\n','\n')
        
        request = {}
        for col in df_data_tmp.columns:
            request[col] = df_data_tmp[col][k]

        #response = get_model_response(prompt_indiv)
        request["model"] = "gpt-3.5-turbo"
        request["messages"] = [{"role": "user", "content":prompt_indiv}]
        request["max_tokens"] = 256

        all_requests.append(request)
    
    #print(all_requests)
    start = datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(send_request, all_requests)
    end = datetime.now()

    # sleep 5 mintues to avoid the rate limit
    print("Time elapsed:", end - start)

    print("=="* 20)

    #time.sleep(300)  ###############################
    

