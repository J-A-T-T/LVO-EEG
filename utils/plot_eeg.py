import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd

def plot_eeg(data):
    sfreq = 220  # Hz
    # FH = np.load("./data/processed_eeg.npy")
    # FH = FH[0]
    FH = data
    print(FH.shape)
    channel_names = "TP9 AF7 AF8 TP10".split()
    channel_types = "eeg eeg eeg eeg".split()
    # Create the info structure needed by MNE
    info = mne.create_info(channel_names, sfreq=sfreq, ch_types=channel_types)
    info.set_montage('standard_1020')
    pure_raw = mne.io.RawArray(np.transpose(FH), info)
    raw = pure_raw.copy()
    raw.plot(scalings='auto')


def plot_temp():
    eeg_data = pd.read_csv("./data/feature_processing/001_EEG_baseline_stroke_study_updated.csv")
    plt.figure(figsize=(10, 10))
    plt.rcParams["figure.figsize"] = (10,10)

    for column in eeg_data:
        eeg_data_chosen = eeg_data[column] # choosing second channel 
        x = np.linspace(0, len(eeg_data_chosen), len(eeg_data_chosen))
        plt.plot(x, eeg_data_chosen)
    plt.savefig("./data/eeg_chosen_channel.png")
    plt.show()
    