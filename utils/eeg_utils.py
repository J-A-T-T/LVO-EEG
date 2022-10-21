# from turtle import pd
import pandas as pd
import numpy as np


path = '../data/ecg_clean/'
name_eeg = '_EEG_baseline_stroke_study_updated.csv'
path_acc_gyro = '../data/ACCandGYRO/'
name_acc = '_ACC_baseline_stroke_study_updated.csv'
name_gyro = '_GYRO_baseline_stroke_study_updated.csv'
store = None
count = 0
# length = 35700
length = 5000
for i in range(1,116):
    temp = None
    if i != 110:
        if len(str(i)) == 1:
            eeg = pd.read_csv(path+'00'+str(i)+name_eeg).to_numpy()
            # acc = pd.read_csv(path_acc_gyro+'00'+str(i)+name_acc).to_numpy()
            # gyro = pd.read_csv(path_acc_gyro+'00'+str(i)+name_gyro).to_numpy()

        elif len(str(i)) == 2:
            eeg = pd.read_csv(path+'0'+str(i)+name_eeg).to_numpy()
            # acc = pd.read_csv(path_acc_gyro+'0'+str(i)+name_acc).to_numpy()
            # gyro = pd.read_csv(path_acc_gyro+'0'+str(i)+name_gyro).to_numpy()
        else:
            eeg = pd.read_csv(path+str(i)+name_eeg).to_numpy()
            # acc = pd.read_csv(path_acc_gyro+str(i)+name_acc).to_numpy()
            # gyro = pd.read_csv(path_acc_gyro+str(i)+name_gyro).to_numpy()
    
        eeg = eeg[:length,1:]
        # acc = acc[:length,1:]
        # gyro = gyro[:length,1:]

        # temp  = np.hstack((eeg, acc, gyro))
        temp  = eeg

        temp = temp.reshape(1, temp.shape[0], temp.shape[1])
        if i == 1:
            store = temp
        else:
            store = np.vstack((store, temp))

        count +=1 
print(store.shape)
np.save('../data/processed_eeg.npy', store)

