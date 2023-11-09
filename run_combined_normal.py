import pandas as pd


datasets_list = ['combined_binary_en_convAI.csv', 'combined_binary_en_fb_yt.csv',
                'combined_binary_en_HateCheck.csv', 'combined_binary_en_yt_reddit.csv',
                'EN_combined_binary.csv', ]


#datasets_list = ['combined_binary_en_HateCheck.csv','combined_binary_en_yt_reddit.csv', 'EN_combined_binary.csv',]


df_classifiers = {}
classifiers = ['HateCheck', 'stormfront', 'twitter', 'fb_yt', 'convAI','wikipedia']


from transformers import pipeline

for model in classifiers:
    df_classifiers[model] = pipeline('text-classification', model=f'Output_Dir/{model}', device=0)
    

import pandas as pd
from tqdm import tqdm

df_normal = {}
common_cols = ['text', 'labels', 'binary_labels']
for c in common_cols:
    df_normal[c] = []
for c in classifiers:
    df_normal[c] = []
df_normal['source'] = []

cnt = 4
for dataset in datasets_list[-1:]:
    df_data = pd.read_csv(f'Datasets/{dataset}')
    df_data = df_data[df_data['binary_labels']==0] ##############
    df_data = df_data.dropna(subset=['text','binary_labels'])
    df_data = df_data.reset_index(drop=True,inplace=False)
    
    
    text_list = df_data['text'].tolist()
    for ind,text in enumerate(tqdm(text_list)):
        try:
            for model in classifiers:
                pred = df_classifiers[model](text)
                df_normal[model].append(pred[0]['label'])
        except:
            continue
            
        for col in common_cols:
            #tmp_list = df_data[col].tolist()
            df_normal[col].append(df_data[col][ind])
        df_normal['source'].append(dataset)
    
    
            
        cnt+=1
        if(cnt%1000==0):
          df_tmp = pd.DataFrame(df_normal)
          df_tmp.to_csv(f'Datasets/Normal_1/Combined/All_Normal_{cnt}.csv',index=False)

df_normal = pd.DataFrame(df_normal)
df_normal.to_csv(f'Datasets/Normal_1/All_Normal_Total_1.csv',index=False)

print(len(df_normal['text']))


df_0 = pd.read_csv(f'Datasets/Normal_1/All_Normal_Total_0.csv')
df_1 = pd.read_csv(f'Datasets/Normal_1/All_Normal_Total_1.csv')

df_2 = pd.concat([df_0,df_1], ignore_index=True)

df_2.to_csv(f'Datasets/Normal_1/All_Normal_Total.csv',index=False)