import warnings
import matplotlib
warnings.filterwarnings('ignore')
import sklearn
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
        warnings.simplefilter('ignore')
warnings.simplefilter("ignore", category=ConvergenceWarning)
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import RepeatedStratifiedKFold


def model(eeg_features, feature_extraction_method):

    clinical_features = pd.read_csv(r'./data/df_onsite.csv')
    lvo = clinical_features['lvo']
    clinical_features = clinical_features.drop(['lvo'], axis=1)

    all_features = pd.concat([eeg_features, clinical_features], axis=1)

    gp = GaussianProcessClassifier()
    # define grid
    params = dict()
    params['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel(), 1.0 * ExpSineSquared()]
    #custom_scorer = {'ACC': make_scorer(acc, greater_is_better=True), 'Custom Evaluation':make_scorer(evaluation_metric, greater_is_better=False)}

    # Internal CV
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(gp, params, scoring='accuracy', cv=inner_cv)

    #custom_scorer = {'ACC': make_scorer(acc), 'Custom Evaluation':make_scorer(evaluation_metric)}

    # External CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_score = cross_validate(grid, X=all_features, y=lvo, cv=outer_cv, scoring="accuracy")
    
    print(nested_score)
    print(nested_score['test_score'].mean())

def normalize_data(eeg_features, feature_extraction_method):
    if feature_extraction_method == 'simple':
    
        # Normalize the data using z-score
        scaler = StandardScaler()
        scaler.fit(eeg_features)
        transformer = FunctionTransformer(zscore)
        eeg_features_ = transformer.transform(eeg_features)

        eeg_features = pd.DataFrame(eeg_features_, columns = eeg_features.columns)
    
    elif feature_extraction_method == 'wavelet':

        # Normalize the data
        scaler = StandardScaler()
        scaler.fit(eeg_features)
        eeg_features_ = scaler.transform(eeg_features)

        eeg_features = pd.DataFrame(eeg_features_, columns = eeg_features.columns)

    return eeg_features
    

def evaluation_metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    expected_loss = (4*FN+FP)/(4*(TP+FP)+(TN+FN))
    return expected_loss


if __name__ == '__main__':
    
    eeg_features1 = pd.read_csv(r'data\feature_processed\simple_features.csv')
    gyro_features = pd.read_csv(r'data\feature_processed\simple_gyro_features.csv')
    acc_features = pd.read_csv(r'data\feature_processed\simple_acc_features.csv')
    eeg_features2 = pd.read_csv(r'data\feature_processed\features.csv')

    x = input("""Please select the option for the preprocessed the EEG data:
    1. Simple Feature Exatraction
    2. Wavelet Feature Extraction \n
    """)

    if int(x) == 1:
        eeg_features = normalize_data(eeg_features1, 'simple')
    elif int(x) == 2:
        eeg_features = normalize_data(eeg_features2, 'wavelet')
    else:
        x = input("Please enter 1 or 2")

    gyro_features = normalize_data(gyro_features, 'simple')
    acc_features = normalize_data(acc_features, 'simple')

    all_eeg_features = pd.concat([eeg_features, gyro_features, acc_features], axis=1)

    if int(x) == 1:
        model(eeg_features, 'simple')
    elif int(x) == 2:
        model(eeg_features, 'wavelet')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')