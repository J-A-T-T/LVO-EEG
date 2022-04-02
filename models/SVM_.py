import warnings
import matplotlib
from sklearn.gaussian_process import GaussianProcessClassifier
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

def model(eeg_features, feature_extraction_method):

    clinical_features = pd.read_csv(r'./data/df_onsite.csv')
    lvo = clinical_features['lvo']
    clinical_features = clinical_features.drop(['lvo'], axis=1)

    all_features = pd.concat([eeg_features, clinical_features], axis=1)

    # define grid
    params = [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10, 100]
        },

        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10, 100],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        },

        {
            "kernel": ["poly"],
            "C": [0.1, 1, 10, 100],
            "degree": [2, 3, 4, 5]
        },

        {
            "kernel": ["sigmoid"],
            "C": [0.1, 1, 10, 100],
        }
    
    ]

    # request probability estimation
    svm = SVC(probability=True, max_iter=100000)
    custom_scorer = {'ACC': make_scorer(acc, greater_is_better=True), 'Custom Evaluation':make_scorer(evaluation_metric, greater_is_better=False)}

    # Internal CV
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(svm, params, scoring=custom_scorer, cv=5, refit='ACC')

    grid.fit(all_features, lvo)
    print('\n All results:')
    print(grid.cv_results_)
    print('\n Best estimator:')
    print(grid.best_estimator_)
    print('\n Best score:')
    print(grid.best_score_ * 2 - 1)
    print('\n Best parameters:')
    print(grid.best_params_)
    results = pd.DataFrame(grid.cv_results_)
    results.to_csv('./results/svm-grid-search.csv', index=False)

    custom_scorer = {'ACC': make_scorer(acc), 'Custom Evaluation':make_scorer(evaluation_metric)}

    # External CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_score = cross_validate(grid, X=all_features, y=lvo, cv=outer_cv, scoring=custom_scorer)
    
    print(nested_score)
    print(nested_score['test_ACC'].mean())
    print(nested_score['test_Custom Evaluation'].mean())

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

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', SVC(probability=True, C= 1, kernel = 'linear', max_iter=10000)))
	#level0.append(('cart', RandomForestClassifier()))
	level0.append(('svm', SVC(probability=True, C= 0.1, degree = 2, kernel = 'poly', max_iter=10000)))
	#level0.append(('bayes', GaussianProcessClassifier()))
	# define meta learner model
	level1 = SVC()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
	return model

def evaluate_model(model, X, y):
    custom_scorer = {'ACC': make_scorer(acc), 'Custom Evaluation':make_scorer(evaluation_metric)}

    # External CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    nested_score = cross_validate(model, X=X, y=y ,cv=outer_cv, scoring=custom_scorer)
    return nested_score

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

    eeg_features1 = normalize_data(eeg_features1, 'simple')
    #eeg_features2 = normalize_data(eeg_features2, 'wavelet')

    gyro_features = normalize_data(gyro_features, 'simple')
    acc_features = normalize_data(acc_features, 'simple')

    all_eeg_features = pd.concat([eeg_features1, gyro_features, acc_features], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

    clinical_features = pd.read_csv(r'./data/df_onsite.csv')
    lvo = clinical_features['lvo']
    clinical_features = clinical_features.drop(['lvo'], axis=1)

    all_features = pd.concat([all_eeg_features, clinical_features], axis=1)

    # Best approach was combining the data

    model(eeg_features1,"simple")

    # 1 approach is voting classifier
    # model1 = SVC(probability=True, C= 0.1, degree = 2, kernel = 'poly')
    # model2 = SVC(probability=True, C= 1, kernel = 'linear')
    # model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
    # model.fit(X_train,y_train)
    # score = model.score(X_test,y_test)
    # print(score)

    # Another approach is stacking 
    # get_stacking().fit(X_train, y_train)
    # scores = evaluate_model(get_stacking(), all_features ,lvo)
    # acc = scores['test_ACC'].mean()
    # loss = scores['test_Custom Evaluation'].mean()
    # print('Accuracy: ', acc)
    # print('Loss: ', loss)