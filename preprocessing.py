import numpy as np
import mne as mne

class Preprocessing():
    def __init__(self):
        pass

    def save_data(self, filename, data):
        """
        Save our array to a file.
        """
        x = filename.split("\\")
        name = x[-1]
        x = x[:-2]
        x = "\\".join(x)
        x += "\ecg_clean\\"
        x += name
        data.to_csv(x, index=False)


    def load_in_data(self, bigarray):
        """
        Loads in the data.
        :param filename:
        :return:
        """
        sfreq = 220  # Hz
        print(bigarray.shape)
        channel_names = "TP9 AF7 AF8 TP10 Right_Aux Marker".split()
        channel_types = "eeg eeg eeg eeg ecg stim".split()
        # Create the info structure needed by MNE
        info = mne.create_info(channel_names, sfreq=sfreq, ch_types=channel_types)
        info.set_montage('standard_1020')
        pure_raw = mne.io.RawArray(np.transpose(bigarray), info)
        return pure_raw


    def preprocess(self, data):
        """
        Preprocesses the data.
        :param data:
        :return:
        """
        if "EEG" not in data:
            return None
        with open(data, 'r') as f:
            lines = (line for line in f if not line.startswith('#'))
            bigarray = np.loadtxt(lines, delimiter=',', skiprows=1)
        # removing our first column timestamp
        pure_raw = np.delete(bigarray, 0,1)
        # FH = np.delete(FH, 5,1)
        # FH = np.delete(FH, 4,1)
        # Sampling rate
        
        # Create the mne object
        raw = self.load_in_data(pure_raw.copy())

        raw.pick_types(meg=False, eeg=True, ecg=False, stim=True)
        raw_copy = raw.copy()
        filt_raw = raw_copy.filter(filter_length="auto",l_freq=.1, h_freq=15)
        events = mne.find_events(filt_raw, stim_channel='Marker')
        epochs = mne.Epochs(filt_raw, events, event_id=None, tmin=-0.2, tmax=0.5)
        ica = mne.preprocessing.ICA(n_components=4, method='fastica').fit(epochs)
        ecg_evoked = mne.preprocessing.create_ecg_epochs(pure_raw.copy()).average()
        ecg_evoked.apply_baseline(baseline=(None, -0.2))
        ecg_evoked.plot_joint(title='ECG epochs')
        ica.exclude = []
        raw = pure_raw.copy()
        # find which ICs match the ECG pattern
        ecg_indices, ecg_scores = ica.find_bads_ecg(pure_raw, method='correlation',
                                                    threshold='auto')
        ica.exclude = ecg_indices

        # barplot of ICA component "ECG match" scores
        ica.plot_scores(ecg_scores)

        print(ecg_indices)
        # plot diagnostics
        ica.plot_properties(pure_raw, picks=[3])

        # plot ICs applied to raw data, with ECG matches highlighted
        ica.plot_sources(filt_raw, show_scrollbars=False)

        # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
        ica.plot_sources(ecg_evoked)
        ica.exclude = [0]
        ica.fit(filt_raw, picks=p, decim=3)
