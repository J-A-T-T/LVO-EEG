
import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer
import shap
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import LabelEncoder

# References : https://stackoverflow.com/questions/45074579/votingclassifier-different-feature-sets

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):
    
    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    # Fit all estimators with their respective feature arrays
    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return label_encoder.inverse_transform(pred)


def train_svm_eeg(X_train, X_test, y_train, y_test):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    

    param = [
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
    svm = SVC(probability=True)

    custom_scorer = {'ACC_': make_scorer(acc, greater_is_better=True), 'ExpectedLoss':make_scorer(evaluation_metric, greater_is_better=False)}

    #custom_scorer = make_scorer(acc, greater_is_better=True)
    #custom_scorer = make_scorer(evaluation_metric, greater_is_better=False)
    # 5-fold cross validation
    clf = GridSearchCV(svm, param,
            cv=5, verbose=3, scoring=custom_scorer, n_jobs=-1, refit='ACC_')

    clf.fit(X_train, y_train)

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict= clf.best_estimator_.predict(X_test)
    y_predict_train = clf.best_estimator_.predict(X_train)

    print('\n All results:')
    print(clf.cv_results_)
    print('\n Best estimator:')
    print(clf.best_estimator_)
    print('\n Best score:')
    print(clf.best_score_ * 2 - 1)
    print('\n Best parameters:')
    print(clf.best_params_)

    print("Train Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_train, y_predict_train)))
    print("Test Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_test, y_predict)))

    print("\nLoss Metric:")

    eval = evaluation_metric(y_test, y_predict)  
    eval_train = evaluation_metric(y_train, y_predict_train)

    print("Train Loss : {0}".format(eval_train))
    print("Test Loss : {0}".format(eval))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

    model = clf

    return model

def train_svm_clinical(X_train, X_test, y_train, y_test):

    linear_svc = SVC(kernel='linear', C=1.0, probability=True).fit(X_train, y_train)
    rbf_svc = SVC(kernel='rbf', C=1.0, probability=True).fit(X_train, y_train)
    poly_svc = SVC(kernel='poly', C=1.0, probability=True).fit(X_train, y_train)
    sigmoid_svc = SVC(kernel='sigmoid', C=1.0, probability=True).fit(X_train, y_train)

    linear_svc_pred = linear_svc.predict(X_test)
    rbf_svc_pred = rbf_svc.predict(X_test)
    poly_svc_pred = poly_svc.predict(X_test)
    sigmoid_svc_pred = sigmoid_svc.predict(X_test)

    linear_svc_eval = evaluation_metric(y_test, linear_svc_pred)
    rbf_svc_eval = evaluation_metric(y_test, rbf_svc_pred)
    poly_svc_eval = evaluation_metric(y_test, poly_svc_pred)
    sigmoid_svc_eval = evaluation_metric(y_test, sigmoid_svc_pred)

    linear_svc_F1  = sklearn.metrics.accuracy_score(y_test,linear_svc_pred )
    rbf_svc_F1  = sklearn.metrics.accuracy_score(y_test,rbf_svc_pred )
    poly_svc_F1 = sklearn.metrics.accuracy_score(y_test,poly_svc_pred)
    sigmoid_svc_F1  = sklearn.metrics.accuracy_score(y_test,sigmoid_svc_pred )

    model = linear_svc   # has the highest accuracy and lowest loss

    print("\nLinear SVM:")
    print("Train Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_train, linear_svc.predict(X_train))))
    print("Test Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_test, linear_svc.predict(X_test))))

    return model

def combine(svm1, svm2, X_train, y_train, X_test, y_test):

    ensemble=VotingClassifier(estimators=[('SVM', svm1), ('SVM2', svm2)], 
                       voting='soft', weights=[1,1]).fit(X_train,y_train)
    print('The accuracy for combination is :',ensemble.score(X_test, y_test))
    

def svm_clinical():
    df = pd.read_csv('./data/df_onsite.csv')
    lvo = df['lvo']
    features = df.drop(['lvo'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, lvo, test_size=0.2, random_state=42)

    model = train_svm_clinical(X_train, X_test, y_train, y_test)


def svm_eeg():
    # Load the datasets

    clinical_features = pd.read_csv('./data/df_onsite.csv')
    lvo = clinical_features['lvo']
    clinical_features = clinical_features.drop(['lvo'], axis=1)

    eeg_features = pd.read_csv(r'data\feature_processed\simple_features.csv')

    # Normalize the data using z-score
    scaler = StandardScaler()
    scaler.fit(eeg_features)
    transformer = FunctionTransformer(zscore)
    eeg_features_ = transformer.transform(eeg_features)

    eeg_features = pd.DataFrame(eeg_features_, columns = eeg_features.columns)

    X_train, X_test, y_train, y_test = train_test_split(eeg_features, lvo, test_size=0.2, random_state=42)

    model = train_svm_eeg(X_train, X_test, y_train, y_test)
    
    return model



def combine_eeg_clinical_models():
    # Load the datasets
    clinical_features = pd.read_csv('./data/df_onsite.csv')
    lvo = clinical_features['lvo']
    clinical_features = clinical_features.drop(['lvo'], axis=1)

    eeg_features = pd.read_csv(r'data\feature_processed\simple_features.csv')

    # Normalize the data using z-score
    scaler = StandardScaler()
    scaler.fit(eeg_features)
    transformer = FunctionTransformer(zscore)
    eeg_features_ = transformer.transform(eeg_features)

    eeg_features = pd.DataFrame(eeg_features_, columns = eeg_features.columns)

    # Combine eeg and clinical features
    all_features = pd.concat([eeg_features, clinical_features], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(all_features, lvo, test_size=0.2, random_state=42)

    # Split eeg and clinical training and testing sets for different models
    X_train_eeg = X_train[X_train.columns[0:8]]
    X_train_clinical = X_train[X_train.columns[8:]]
    X_test_eeg = X_test[X_test.columns[0:8]]
    X_test_clinical = X_test[X_test.columns[8:]]

    X_train_list = [X_train_eeg, X_train_clinical]
    X_test_list = [X_test_eeg, X_test_clinical]
    
    # The models
    classifiers = [('svc1', SVC(kernel='poly', C=0.1, degree=2, probability=True)),
    ('svc2', SVC(kernel="linear", C=1, probability=True))]

    fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train)

    y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list)
    y_pred_train = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_train_list)

    print("Train Accuracy and Loss:")
    print(accuracy_score(y_train, y_pred_train))
    print(evaluation_metric(y_train, y_pred_train))

    print("Test Accuracy and Loss:")
    print(accuracy_score(y_test, y_pred))
    print(evaluation_metric(y_test, y_pred))


def evaluation_metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    expected_loss = (4*FN+FP)/(4*TP+TN)
    return expected_loss

def acc(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)

 
if __name__ == '__main__':
  
    #eeg_model = svm_eeg()
    #clinical_model = svm_clinical()

    combine_eeg_clinical_models()

    
    