import warnings
import matplotlib
warnings.filterwarnings('ignore')

from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
        warnings.simplefilter('ignore')
warnings.simplefilter("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
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

    X_train, X_test, y_train, y_test = train_test_split(all_features, lvo, test_size=0.2, random_state=42)


    gp = GaussianProcessClassifier()
    # define grid
    params = dict()
    params['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel(), 1.0 * ExpSineSquared()]
    custom_scorer = {'ACC': make_scorer(acc, greater_is_better=True), 'Custom Evaluation':make_scorer(evaluation_metric, greater_is_better=False)}
    #custom_scorer = make_scorer(acc, greater_is_better=True)

    # define search
    grid = GridSearchCV(gp, params, scoring=custom_scorer, cv=5, refit='ACC')

    grid.fit(X_train, y_train)
    print('\n All results:')
    print(grid.cv_results_)
    print('\n Best estimator:')
    print(grid.best_estimator_)
    print('\n Best score:')
    print(grid.best_score_ * 2 - 1)
    print('\n Best parameters:')
    print(grid.best_params_)
    results = pd.DataFrame(grid.cv_results_)
    results.to_csv('./results/gp-grid-search.csv', index=False)

    y_predict= grid.best_estimator_.predict(X_test)
    y_predict_train = grid.best_estimator_.predict(X_train)

    print("Train Accuracy: {0}".format(accuracy_score(y_train, y_predict_train)))
    print("Test Accuracy: {0}".format(accuracy_score(y_test, y_predict)))

    print("\nCustom Evaluation:")

    eval = evaluation_metric(y_test, y_predict)  
    eval_train = evaluation_metric(y_train, y_predict_train)

    print("Train Custom Evaluation Metric : {0}".format(eval_train))
    print("Test Custom Evaluation Metric : {0}".format(eval))

    test_probs = grid.predict_proba(X_test)[:,1]
    print(test_probs)
   


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


def acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
    

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
    eeg_features2 = pd.read_csv(r'data\feature_processed\features.csv')

    eeg_features1 = normalize_data(eeg_features1, 'simple')
    eeg_features2 = normalize_data(eeg_features2, 'wavelet')

    #model(eeg_features1, 'simple')
    model(eeg_features2, 'wavelet')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')