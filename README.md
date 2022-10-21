# LVO-EEG
A predictor for LVO versus non-LVO

* Data: Email request to Dr. Kyle and Dr. Brian
* Report: [Link](https://drive.google.com/file/d/1Z0c5CyQoB-2tf69kRKkPGwjus9o88jQM/view?usp=sharing)


## Requirements:
pip install -r requirements.txt


# Instruction to run
1. Download the data from the above file
2. Create virtual environment (virtualenv venv -p=python3)
3. Install the requirement package (pip install -r requirements.txt)
4. Run the preprocessing
5. Run the training process


# How to run the Preprocessing 

1. Have a folder data in the home directory within which we have another folder 115 which contains all participants (./data/115)

2. Create another folder called "ecg_clean" in the data folder (./data/ecg_clean)

3. Provide the path to the ./data/115 folder as the variable directory in preprocessing overhead.py

4. Run "py preprocessing_overhead.py" and choose the correct option

5. Wait a minute or so

6. Create another folder called "feature_processing" in the data folder (./data/feature_processing)

7. And a final folder called "feature_processed" ( I know yes another one :( )

*  Dw, they wont take that much space :)

## If it's your first time

* run option 2 first

* Wait a minute or so

* HFD data be in feature_processed and your clean data in feature processing

* run option 4 (acc and gyro should be clean too)

* run option 3 if you want to get your fractal for acc and gyro, edit the code to not overwrite your eeg fractal, or rename the fractal
9. Data should be in ecg clean
7. To train DL/RNN models, run python eeg_utils to obtain processed_eeg.npy

# How to train DL/RNN models
1. Choose a suitable training file
2. train_clinical.py: Train a neural network on clinical data only 
3. train_eegnet_clinical.py: Train a EEGNet on EEG data and a different neural network on clinical data
4. train_lstm_clinical.py: Train a LSTM on EEG data and a different neural network on clinical data
5. train_eegnet.py: Train a EEGNet on EEG data only
6. train_lstm.py: Train a LSTM on EEG data only


# How to train ML models
1. In the root directory run python models/SVM_.py or python models/GP.py or python python models/RF_.py or python models/XGBoost.py  
2. Select the option for the prepreocessed EEG data you want to use

# Results
Please take a look at the table [here](https://docs.google.com/spreadsheets/d/1xypKoyjERM7am8Do1qPqB_lr5vn0G-QxFHnlWegXSWY/edit#gid=0) 

# References:
https://github.com/jordan-bird/eeg-feature-generation
https://eeglib.readthedocs.io/en/latest/

