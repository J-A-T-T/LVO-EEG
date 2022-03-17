from turtle import pd
import pandas as pd
import numpy as np


path = './data/ecg_clean/'
name = '_EEG_baseline_stroke_study_updated.csv'
store = None
count = 0
# length = 42084
length = 120
for i in range(1,84):
    eeg = None
    if len(str(i)) == 1:
        eeg = pd.read_csv(path+'00'+str(i)+name).to_numpy()
    else:
        eeg = pd.read_csv(path+'0'+str(i)+name).to_numpy()
    
    eeg = eeg[:length,1:]
    eeg = eeg.reshape(1, eeg.shape[0], eeg.shape[1])

    if i == 1:
        store = eeg
    else:
        store = np.vstack((store, eeg))

    count +=1 

# for i in range(111, 116):
    #     eeg = pd.read_csv(path+str(i)+name).to_numpy()
    #     eeg = eeg[:length,1:]
    #     eeg = eeg.reshape(1, eeg.shape[0], eeg.shape[1])
    #     store = np.vstack((store, eeg))

    #     count +=1 

np.save('./data/processed_eeg.npy', store)