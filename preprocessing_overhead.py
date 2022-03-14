# import required module
import os

from preprocessing import Preprocessing as pp
# assign directory where the data files are stored
directory = r'C:\Users\tanya\OneDrive\Documents\GitHub\LVO-EEG\data\115'


# create a Preprocessing object
preprocessing = pp()
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)
        # preprocessing.preprocess(f)