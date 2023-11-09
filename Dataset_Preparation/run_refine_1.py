import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm


datasets_list = os.listdir('NEW_Generated_Data/DataFrame/')


#datasets_list = ['combined_binary_en_HateCheck.csv','combined_binary_en_yt_reddit.csv', 'EN_combined_binary.csv',]


df_classifiers = {}
classifiers = ['HateCheck', 'stormfront', 'twitter', 'fb_yt', 'convAI','wikipedia']


for model in classifiers:
    df_classifiers[model] = pipeline('text-classification', model=f'Output_Dir/{model}', device=0)
    
    
    
    

for dataset in tqdm(datasets_list):
    if(not dataset.endswith('.csv')):
        continue
    df_data = pd.read_csv(f'NEW_Generated_Data/DataFrame/{dataset}')
    
    df_new = {}
    for col in df_data.columns:
        df_new[col] = []
    df_new['toxic_count'] = []
    df_new['non_toxic_count'] = []
    for c in classifiers:
        df_new[c] = []
    
    toxic_list = []
    non_toxic_list = []
    for ind in range(len(df_data['data_id'])):
        if(df_data['source_label'][ind]=='toxic'):
            toxic_list.append(df_data['source_text'][ind])
            non_toxic_list.append(df_data['target_text'][ind])
        else:
            non_toxic_list.append(df_data['source_text'][ind])
            toxic_list.append(df_data['target_text'][ind])
        

        for col in df_data.columns:
            df_new[col].append(df_data[col][ind])

    tmp_data = {}


    for model in classifiers:
        tmp_data[f'toxic_{model}'] = df_classifiers[model](toxic_list, max_length = 512, 
                                                           padding = 'max_length', truncation=True)
        tmp_data[f'non_toxic_{model}'] = df_classifiers[model](non_toxic_list, max_length = 512,
                                                               padding = 'max_length', truncation=True)

    for k in range(len(df_data['source_text'])): #zip(tmp_data[f'toxic_{model}'],tmp_data[f'non_toxic_{model}']):
        tox_cnt = non_tox_cnt = 0
        for model in classifiers:
            tox = tmp_data[f'toxic_{model}'][k]['label']
            non_tox = tmp_data[f'non_toxic_{model}'][k]['label']
            df_new[model].append(f'{tox}_{non_tox}')
            tox_cnt+= int(tox)
            non_tox_cnt += int(non_tox)

        df_new['toxic_count'].append(tox_cnt)
        df_new['non_toxic_count'].append(non_tox_cnt)   

    

    df_new = pd.DataFrame(df_new)
    df_new.to_csv(f'NEW_Generated_Data/Pred_DataFrame/{dataset}',index=False)
        
        #print(f'Dataset: {dataset} ## Original Length:{len(df_data['source_text'])} ## New ')
