import json
from itertools import repeat

import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
from data.variables import panels,dags,allocation,errors

with open('/home/pmwaniki/Dropbox/others/Ambrose/Sync study/Benchmark/data/batches_master_index.json', 'r') as f:
    batches=json.load(f)


batches2=pd.concat([pd.DataFrame({'name':repeat(name,len(values)),'study_id':values}) for name,values in batches.items()])
batches2['panel']=batches2['name'].map(allocation)


n_panel=batches2['panel'].value_counts().to_frame()
n_panel['days']=n_panel['count']*(30/np.sum(n_panel['count']))
n_panel=n_panel.reset_index()
n_panel['reviewer0']=n_panel['panel'].map(lambda x: panels[x][0])
n_panel['reviewer1']=n_panel['panel'].map(lambda x: panels[x][1])

n_panel2=pd.melt(n_panel,id_vars=['panel','count','days'],value_vars=['reviewer0','reviewer1'])

final=n_panel2.groupby('value')[['days','count']].agg('sum')








