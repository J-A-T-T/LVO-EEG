import numpy as np
import mne

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