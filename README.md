# LVO-EEG
A predictor for LVO non LVO

* Data https://drive.google.com/drive/folders/1LKz-tQevrR41TiMITG9h7yl7EUoblI16?usp=sharing

* Result https://docs.google.com/spreadsheets/d/1xypKoyjERM7am8Do1qPqB_lr5vn0G-QxFHnlWegXSWY/edit#gid=0



## Requirements:
pip install:
* mne
* gradcam
* pywavelets




# Instruction to run
1. Download the data from the above file
2. Create virtual environment (pipenv)
3. Install the requirement package
4. Run the preprocess
5. Run the training



# How to run the Preprocessing 

* Have a folder data in the home directory within which we have another folder 115 which contains all participants (./data/115)

* Create another folder called "ecg_clean" in the data folder (./data/ecg_clean)

* Create another folder called "feature_processing" in the data folder (./data/feature_processing)

* And a final folder called "feature_processed" ( I know yes another one :( )

* Dw, they wont take that much space :)

* Provide the path to the ./data/115 folder as the variable directory in preprocessing_overhead.py

* Run "py preprocessing_overhead.py"

## If it's your first time

* run option 2 first

* Wait a minute or so

* HFD data be in feature_processed and your clean data in feature processing

* run option 4 (acc and gyro should be clean too)

* run option 3 if you want to get your fractal for acc and gyro, edit the code to not overwrite your eeg fractal, or rename the fractal



# References:
https://github.com/jordan-bird/eeg-feature-generation
https://eeglib.readthedocs.io/en/latest/

