import warnings
import matplotlib
warnings.filterwarnings('ignore')

with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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

def model():
    # A parameter grid for XGBoost
    params = {
        'min_child_weight': [1, 3, 5, 10],
        'gamma': [1, 1.5, 2, 5],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'max_depth': [3, 4, 5],
        'eta': [0.1, 0.2, 0.3]
        }


    clinical_features = pd.read_csv(r'./data/df_onsite.csv')
    lvo = clinical_features['lvo']
    clinical_features = clinical_features.drop(['lvo'], axis=1)

    eeg_features = pd.read_csv(r'data\feature_processed\simple_features.csv')

    # Normalize the data using z-score
    scaler = StandardScaler()
    scaler.fit(eeg_features)
    transformer = FunctionTransformer(zscore)
    eeg_features_ = transformer.transform(eeg_features)

    eeg_features = pd.DataFrame(eeg_features_, columns = eeg_features.columns)

    all_features = pd.concat([eeg_features, clinical_features], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(all_features, lvo, test_size=0.2, random_state=42)

    xgb = XGBClassifier(learning_rate=0.02, objective='binary:logistic', nthread=1, verbosity=0, silent=True)

    custom_scorer = {'ACC': make_scorer(acc, greater_is_better=True), 'ExpectedLoss':make_scorer(evaluation_metric, greater_is_better=False)}
    #custom_scorer = make_scorer(acc, greater_is_better=True)
    grid = GridSearchCV(estimator=xgb, param_grid=params, scoring=custom_scorer, n_jobs=4, cv=5, verbose=3, refit='ACC' )
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
    results.to_csv('./results/xgb-grid-search.csv', index=False)

    y_predict= grid.best_estimator_.predict(X_test)
    y_predict_train = grid.best_estimator_.predict(X_train)

    print("Train Accuracy: {0}".format(accuracy_score(y_train, y_predict_train)))
    print("Test Accuracy: {0}".format(accuracy_score(y_test, y_predict)))

    print("\nLoss Metric:")

    eval = evaluation_metric(y_test, y_predict)  
    eval_train = evaluation_metric(y_train, y_predict_train)

    print("Train Loss : {0}".format(eval_train))
    print("Test Loss : {0}".format(eval))

    test_probs = grid.predict_proba(X_test)[:,1]
    print(test_probs)
   

    #submission = pd.DataFrame({"id": X_test["id"], "prediction": test_probs})
    #submission.to_csv("xgboost_best_parameter_submission.csv", index=False)
    #results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
    #results_df.to_csv('submission-grid-search-xgb-porto-01.csv', index=False)
    #feature_imp = grid.best_estimator_.feature_importance()
    feature_imps = grid.best_estimator_.feature_importances_
    indices = np.argsort(feature_imps)
    feat_importances = pd.Series(feature_imps, index=X_train.columns)
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(X_train.shape[1]), feature_imps[indices],
       color="r", align="center")


    plt.bar( range(len(feature_imps)), feature_imps)
    plt.xticks(range(len(feature_imps)), X_train.columns)
    plt.show()

    

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
    
    model()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')