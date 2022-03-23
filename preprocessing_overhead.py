# import required module
import os

from sklearn import preprocessing

from preprocessing import Preprocessing as pp
# assign directory where the data files are stored
directory = r'C:\Users\tanya\OneDrive\Documents\GitHub\LVO-EEG\data\115'


# create a Preprocessing object
preprocessing = pp()
preprocessing.createSimpleExtractionCSV(directory)
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         returned_df = preprocessing.preprocess(f)
#         if returned_df is not None:
#             preprocessing.save_data(f, returned_df)