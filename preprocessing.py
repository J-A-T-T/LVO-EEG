import numpy as np
import pandas as pd
import mne as mne
import eeglib as eeg
import os

class Preprocessing():
    def __init__(self):
        self.visited = set()

    def save_data(self, filename, data):
        """
        Save our array to a file.
        Saves to _EEG_baseline_stroke_study_updated.csv
        can be changed to save to same file name
        """
        x = filename.split("\\")
        name = x[-1]
        x = x[:-2]
        x = "\\".join(x)
        x += "\ecg_clean\\"
        y = (filename.split("\\")[-1][:3])
        name = y +'_EEG_baseline_stroke_study_updated.csv'
        x += name
        x.replace("\\" , "/")
        print(x)
        
        
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
    
    def simpleExtractionMotion(self, filename, acc=True):
        # load in the data
        df = pd.read_csv(filename)

        #remove timestamps and sequence ID
        if len(df.columns)<=5:#Other format see file 99
            df.drop(df.columns[[0,-1]], axis=1, inplace=True)
        else: #typical format
            df.drop(df.columns[[0, 1,-1]],axis=1,inplace=True)
        if acc:
            remainder = "_ACC_baseline_stroke_study_updated.csv"
        else:
            remainder = "_GYRO_baseline_stroke_study_updated.csv"
        y = "data/feature_processing/" + filename.split("\\")[-1][:3] + remainder
        df.to_csv(y, index=False)
        # helper = eeg.helpers.CSVHelper(y, lowpass=30, highpass=0.5, normalize=True, ICA=True, windowSize=128)
        
        
        helper = eeg.helpers.CSVHelper(y)
        wrap = eeg.wrapper.Wrapper(helper)

        for egg in helper:
            egg.PFD()
        wrap.addFeature.HFD()
        wrap.addFeature.DFA()
        features = wrap.getFeatures()
        tiny = list(features)
        tiny.pop()
        return tiny


    def simpleExtraction(self, filename):
        """
        Extracts simple features from file and places the data
        in a csv
        :param filename:
        :return:
        """
        # load in the data
        df = pd.read_csv(filename)
        df.drop(df.columns[[-1, -2]],axis=1,inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        y = "data/feature_processing/" + filename.split("\\")[-1][:3] + "_EEG_baseline_stroke_study_updated.csv"
        df.to_csv(y, index=False)
        helper = eeg.helpers.CSVHelper(y, lowpass=30, highpass=0.5, normalize=True, ICA=True, windowSize=128)
        wrap = eeg.wrapper.Wrapper(helper)
        for egg in helper:
            egg.PFD()
        wrap.addFeature.HFD()
        wrap.addFeature.DFA()
        features = wrap.getFeatures()
        tiny = list(features)
        tiny.pop()
        return tiny
        

    def createSimpleExtractionCSVEEG(self, directory):
        """
        Creates a csv file for each participant
        :param filename:
        :return:
        """
        self.visited = set()
        new_df = pd.DataFrame(columns=("HFD0", "HFD1", "HFD2", "HFD3", "DFA0", "DFA1", "DFA2", "DFA3"))
        # create a Preprocessing object
        i = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                tiny_key = int(f.split("\\")[-1][:3])
                if "EEG_baseline" not in f and "EEG_post" not in f or tiny_key in self.visited:
                    continue
                returned_list = self.simpleExtraction(f)
                if returned_list is not None:
                    new_df.loc[i] = returned_list
                    i += 1
                    self.visited.add(tiny_key)
                    print(i, tiny_key)
        new_df.to_csv("data/feature_processed/simple_features.csv", index=False)
        

    def createSimpleExtractionCSVACC(self, directory):
        self.visited = set()
        new_df = pd.DataFrame(columns=("HFD0", "HFD1", "HFD2", "DFA0", "DFA1", "DFA2"))
        # create a Preprocessing object
        i = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                tiny_key = int(f.split("\\")[-1][:3])
                if "ACC_baseline" not in f and "ACC" not in f or tiny_key in self.visited:
                    continue
                returned_list = self.simpleExtractionMotion(f, acc=True)
                print(len(returned_list), returned_list)
                if returned_list is not None:
                    new_df.loc[i] = returned_list
                    i += 1
                    self.visited.add(tiny_key)
                    print(i, tiny_key)
        new_df.to_csv("data/feature_processed/simple_acc_features.csv", index=False)


    def createSimpleExtractionCSVGYRO(self, directory):
        self.visited = set()
        new_df = pd.DataFrame(columns=("HFD0", "HFD1", "HFD2",  "DFA0", "DFA1", "DFA2"))
        # create a Preprocessing object
        i = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                tiny_key = int(f.split("\\")[-1][:3])
                if "GYRO_baseline" not in f and "GYRO" not in f or tiny_key in self.visited:
                    continue
                returned_list = self.simpleExtractionMotion(f, acc=False)
                if returned_list is not None:
                    new_df.loc[i] = returned_list
                    i += 1
                    self.visited.add(tiny_key)
                    print(i, tiny_key)
        new_df.to_csv("data/feature_processed/simple_gyro_features.csv", index=False)


    def preprocess(self, data):
        """
        Preprocesses the data.
        :param data:
        :return: preprocessed data
        """
        tiny_key = int(data.split("\\")[-1][:3]) # get the participant number
        
        # if tiny_key in self.visited:
        #     print(tiny_
        # key)
        #     return None
        # self.visited.add(tiny_key)
        if "EEG_baseline" not in data and "EEG_post" not in data or tiny_key in self.visited:
            return None
        
        with open(data, 'r') as f:
            lines = (line for line in f if not line.startswith('#'))
            bigarray = np.loadtxt(lines, delimiter=',', skiprows=1)
        # removing our first column timestamp
        pure_raw = np.delete(bigarray, 0,1)

        # Create the raw object
        raw = self.load_in_data(pure_raw.copy())

        # Filter the data
        raw.pick_types(meg=False, eeg=True, ecg=False, stim=True)
        raw_copy = raw.copy()
        filt_raw = raw_copy.filter(filter_length="auto",l_freq=.1, h_freq=15)

        # Find events with stim channel
        events = mne.find_events(filt_raw, stim_channel='Marker')
        epochs = mne.Epochs(filt_raw, events, event_id=None, tmin=-0.2, tmax=0.5)


        ica = mne.preprocessing.ICA(n_components=3, method='fastica').fit(epochs)
        ica.exclude = [0]
        done = ica.apply(filt_raw)

        # ica = mne.preprocessing.ICA(n_components=4,method='fastica', random_state=23, max_iter=800).fit(filt_raw)
        # p = "eeg"
        # ica.exclude =[3]
        # raw_clean = ica.apply(filt_raw.copy())
        # filt_raw.pick_types(meg=False, eeg=True, ecg=False, stim=False)
        
        
        
        # filt_raw.pick_types(meg=False, eeg=True, ecg=False, stim=False)
        # ica.exclude =[2]
        # print(ica.exclude)
        # done = ica.apply(filt_raw)
        done.pick_types(meg=False, eeg=True, ecg=False, stim=False)
        self.visited.add(tiny_key)
        return done.to_data_frame()
