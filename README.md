# LVO-EEG
A predictor for LVO non LVO

* Data https://drive.google.com/drive/folders/1LKz-tQevrR41TiMITG9h7yl7EUoblI16?usp=sharing

Install mne:
    pip install mne

# How to run the Preprocessing 

* Have a folder data in the home directory within which we have another folder 115 which contains all participants (./data/115)

* Create another folder called "ecg_clean" in the data folder (./data/ecg_clean)

* Provide the path to the ./data/115 folder as the variable directory in preprocessing overhead.py

* Run "py preprocessing_overhead.py"

* Wait a minute or so

* data should be in ecg clean

