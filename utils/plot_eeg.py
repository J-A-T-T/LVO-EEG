import matplotlib.pyplot as plt
import numpy as np
import mne
import matplotlib.pyplot as plt

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
    # raw.plot(scalings='auto')
    # plt.axis('off')
    # plt.show()

    # raw_selection = raw['TP9', 0:5000]
    # x = raw_selection[1]
    # y = raw_selection[0].T
    # plt.plot(x,y)
    # plt.show()
    df = raw.to_data_frame(picks=['eeg'], start=0, stop=5000)
    # then save using df.to_csv(...), df.to_hdf(...), etc
    print(df)
    exit()
    plt.savefig('eeg.png')
