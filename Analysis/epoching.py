import numpy as np
import pandas as pd
from os import listdir

def get_segments(path):
    df_eeg = pd.read_csv(path+'/Subject_'+path[-1]+'_ultracortex.csv', index_col=0)
    data = pd.read_csv(path+'/Subject_'+path[-1]+'_aux.csv', index_col=0)
    data=data.to_numpy()
    columns=df_eeg.columns
    eeg_freq=125
    eeg_sample=[]
    for el in data:
        if float(el[-1])==-1.0:
            eeg_sample.append(el[1])
    segments=[]
    for column in columns:
        seg=[]
        for sample in eeg_sample:
            seg.append(df_eeg.loc[sample-1:sample-1+25*eeg_freq].to_numpy)
        segments.append(seg)
    return segments

path = 'D:/Downloads 2/Andre_Thesis/Data/WOLF_dataset/'
folders=listdir(path)
output=[]
if folders.__contains__('.DS_Store'):
    folders.remove('.DS_Store')
for i in range(len(folders)):
    output.append([])
for folder in folders:
    eeg_emotibit_seg=get_segments(path+folder)
    output[int(folder[-1])]=eeg_emotibit_seg
output=np.array(output,dtype=object)