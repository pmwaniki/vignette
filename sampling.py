
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env")

data_folder=Path(os.getenv("DATA_FOLDER"))

data=pd.read_excel(data_folder / 'Consolidated Dataset V7.xlsx')
panels=data['Clinical Panel'].unique().tolist()

panels_map = {
    'Adult Health':['adult health', 'Adult Health - ENT', 'Adult Health - Medical', 'adult health', 'Adult Health - Opthalmology'],
    'Child Health':['Child Health', 'Child Health - ENT'],
    'General Emergency':['General Emergency', 'Emergency Care - Adult', 'Emergency Care - Burns', 'Emergency Care - ENT'],
    'Maternal and Child Health':['Maternal and Child Health', 'Emergency Care - MCH'],
    'Emergency Care - Pediatric':['Emergency Care - Pediatric', 'Emergency Care - Pediatric Burns'],
    'Surgical Care':['Surgical Care'],
    'Mental Health':['Mental Health', 'Mental Health - Pediatrics', 'Emergency Care - Mental Health'],
    'Neonatal Care':['Neonatal Care'],
    'Critical Care':['Critical Care', 'CRITICAL CARE'],
    'Emergency Care - GBV':['Forensic Case', 'Emergency Care - Rape', 'Emergency Care - GBV', 'Emergency Care - Pediatric Rape', 'Emergency Care - Pediatric GBV'],
    'Sexual And Reproductive Health':['Sexual And Reproductive Health'],
    'Palliative Care':['Palliative Care', 'PALLIATIVE CARE']


}

panel_map2=[{i:k for i in v} for k,v in panels_map.items() ]
panel_map3={}
for d in panel_map2:
    for k,v in d.items():
        panel_map3[k]=v

data['Category']=data['Nursing Competency'].replace(panel_map3)

# d1=data['Clinical Panel'].value_counts().to_frame(name="Frequency").reset_index()
# d2=data['Nursing Competency'].value_counts().to_frame(name="Frequency").reset_index()

n_per_group=np.ceil(500/12)


group_freq=data['Category'].value_counts().to_frame(name="Frequency").reset_index()

sampled_small_groups=data.loc[data["Category"].isin(['Critical Care','Palliative Care'])].copy()

remaining_n=500-len(sampled_small_groups)
remaining_categories  = np.setdiff1d(data['Category'].unique(), sampled_small_groups['Category'].unique())
remaining_categories=sorted(remaining_categories)
n_per_group=int(np.ceil(remaining_n/len(remaining_categories)))
rng = np.random.RandomState(123)

sampled_records=[]
for g in remaining_categories:
    sub=data.loc[data['Category']==g].copy().reset_index(drop=True)
    sampled_sub=sub.iloc[rng.permutation(n_per_group)]
    sampled_records.append(sampled_sub)


final_sample=pd.concat(sampled_records+[sampled_small_groups],axis=0,ignore_index=True)
assert final_sample['Master_Index'].duplicated().sum()==0, "Duplicates found"

final_sample['Category'].value_counts()

final_sample.to_csv(data_folder / 'final_sample.csv',index=False)

