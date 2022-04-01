# import required module
import os

from sklearn import preprocessing

from preprocessing import Preprocessing as pp
# assign directory where the data files are stored
directory = r'C:\Users\tanya\OneDrive\Documents\GitHub\LVO-EEG\data\115'


# create a Preprocessing object
print("What do you want to do")
print("1. Fractal decomposition for EEG")
print("2. clean eeg data")
print("3. Fractal decomposition for ACC and Gyro")
print("4. clean acc and gyro data")
x = int(input("Enter 1,2,3,4\n"))
if x == 1:  
    preprocessing = pp()
    preprocessing.createSimpleExtractionCSVEEG(directory)
elif x == 2:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            returned_df = preprocessing.preprocess(f)
            if returned_df is not None:
                preprocessing.save_data(f, returned_df)
elif x ==3:
    preprocessing = pp()
    preprocessing.createSimpleExtractionCSVACC(directory)
    preprocessing.createSimpleExtractionCSVGYRO(directory)
elif x == 4:
    preprocessing = pp()
    preprocessing.preprocessMotion(directory)
else:
    print("Invalid input")
