from turtle import pd
import pandas as pd
import numpy as np


path = '../data/feature_processing/'
name = '_EEG_baseline_stroke_study_updated.csv'
store = None
count = 0
# length = 35700
length = 5000
for i in range(1,116):
    eeg = None
    if i != 110:
        if len(str(i)) == 1:
            eeg = pd.read_csv(path+'00'+str(i)+name).to_numpy()
        elif len(str(i)) == 2:
            eeg = pd.read_csv(path+'0'+str(i)+name).to_numpy()
        else:
            eeg = pd.read_csv(path+str(i)+name).to_numpy()
    
        eeg = eeg[:length,:]
        eeg = eeg.reshape(1, eeg.shape[0], eeg.shape[1])

        if i == 1:
            store = eeg
        else:
            store = np.vstack((store, eeg))

        count +=1 

np.save('../data/processed_eeg.npy', store)

