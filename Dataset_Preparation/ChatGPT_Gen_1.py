
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
    'toxic': "Input Text: {{$input_text}}\nYour task is text style/attribute transfer. You rewrite the above text into non-toxic language. You must match the target style/attribute and preserve the original meaning as much as possible. You should not need to explain the response. You cannot hallucinate or add anything outside the original input text. You should not include the input text in the response. You should only generate the target text.",
    'non-toxic': "Input Text: {{$input_text}}\nYour task is text style/attribute transfer. You rewrite the above text into toxic language. You must match the target style/attribute and preserve the original meaning as much as possible. You should not need to explain the response. You cannot hallucinate or add anything outside the original input text. You should not include the input text in the response. You should only generate the target text.",
}
LABEL_DICT = {0: "non-toxic", 1:"toxic",}


#df_template = pd.read_excel('Datasets/text-style-transfer.xlsx', sheet_name='all_attribute')
#df_template.head()

df_data = pd.read_csv("Datasets/All_Datasets/NEW_All_dataset_2.csv")
#df_data.head()


def process_response(response):
    response = response.replace('{','')
    response = response.replace('}','')
    response = response.split("Input:")[-1]
    response = response.split("input:")[-1]
    response = response.split("Text:")[-1]
    response = response.split("text:")[-1]
    response = response.strip()
    return response


openai.api_key = open("OPENAI/openai_api_0.txt", "r").readline().strip()
model = "gpt-3.5-turbo"

def send_request(request):
    
    source_text = request.pop("source_text")
    source_label = request.pop("source_label")
    platform = request.pop("platform")
    template = request.pop("template")
    source_file = request.pop("source_file")
    data_id = request.pop("data_id")


    # add retry mechanism when face any issues, such as rate limit. Try 5 times and wait for 2 seconds each time
    for i in range(3):
        try:
            raw_responses = openai.ChatCompletion.create(**request)
            answer = raw_responses["choices"][0]["message"]["content"].replace("{", "").replace("}", "").strip()
            processed_answer = process_response(answer)

            output = {
                "source_text": source_text,
                "target_text": processed_answer, 
                "source_label": source_label, 
                "platform": platform,
                "source_file": source_file,
                "data_id": str(data_id),
                "template": template,
                "raw_responses": raw_responses
            }
            # print(answer, gold_label)
            # print(answer, gold_label)
            with open(result_file, "a") as f:
                json.dump(output, f)
                f.write("\n")
            break
        except Exception as e:
            print(e)
            time.sleep(20)
            
            

            



start = 0


platforms = ['fox news', 'twitter', 'wikipedia', 'twitter and facebook', 'fb_yt', 'reddit', 'yt_reddit', 'facebook', 'hatecheck', 'stormfront', 'gab', 'convai']

for ind in tqdm(range(start, len(platforms))):

    df_platform = df_data[df_data['file_platform']==platforms[ind]]
    #print(len(df_platform['text']))
    df_platform = df_platform.reset_index(drop=True,inplace=False)
    
    df_data_tmp = df_platform.sample(n=min(TOTAL_GEN,len(df_platform)) , replace=False, random_state=None, ignore_index = True)

    cnt = 0
    
    global result_file
    result_file = f"NEW_Generated_Data/{platforms[ind]}_{SAMPLE_SIZE}.jsonl"

    if os.path.exists(result_file):
        os.remove(result_file)
    all_requests = []

    for k in range(len(df_data_tmp['text'])):
        
        source_label = df_data_tmp['binary_labels'][k]
        prompt_template = Template(PROMPTS[LABEL_DICT[source_label]])
        prompt_indiv = prompt_template.substitute(input_text = df_data_tmp['text'][k])
        #prompt_indiv = prompt_indiv.replace('\\n','\n')

        #response = get_model_response(prompt_indiv)
        request = {"model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content":prompt_indiv}],
                'source_text': str(df_data_tmp['text'][k]),
                'source_label': LABEL_DICT[source_label],
                'platform': platforms[ind],
                'template': PROMPTS[LABEL_DICT[source_label]],
                'data_id':df_data_tmp['id'][k],
                'source_file': df_data_tmp['source'][k],
                #"temperature":0,
                "max_tokens":256,
                #"frequency_penalty":0.0,
                #"presence_penalty":0.0,
               }
        all_requests.append(request)
    
    start = datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(send_request, all_requests)
    end = datetime.now()

    # sleep 5 mintues to avoid the rate limit
    print("Time elapsed:", end - start)

    print("=="* 20)

    time.sleep(300)  ###############################

