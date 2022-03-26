
import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer
import shap
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier

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
            "C": [0.1, 1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        },

        {
            "kernel": ["poly"],
            "C": [0.1, 1, 10, 100, 1000],
            "degree": [2, 3, 4, 5]
        },

        {
            "kernel": ["sigmoid"],
            "C": [0.1, 1, 10, 100, 1000],
        }
    
    ]

    # request probability estimation
    svm = SVC(probability=True)

    #ustom_scorer = {'ACC_': make_scorer(acc, greater_is_better=True), 'ExpectedLoss':make_scorer(evaluation_metric, greater_is_better=False)}

    custom_scorer = make_scorer(evaluation_metric, greater_is_better=False)
    # 6-fold cross validation
    clf = GridSearchCV(svm, param,
            cv=5, verbose=3, scoring=custom_scorer, n_jobs=-1)

    clf.fit(X_train, y_train)

    #if os.path.exists(model_output_path):
     #   joblib.dump(clf.best_estimator_, model_output_path)
    #else:
     #   print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict_train = clf.predict(X_train)
    y_predict=clf.predict(X_test)

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

    model = linear_svc

    print("\nLinear SVM:")
    print("Train Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_train, linear_svc.predict(X_train))))
    print("Test Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_test, linear_svc.predict(X_test))))

    return model

def combine(svm1, svm2, X_train, y_train, X_test, y_test):



    ensemble=VotingClassifier(estimators=[('SVM', svm1), ('SVM2', svm2)], 
                       voting='soft', weights=[1,1]).fit(X_train,y_train)
    print('The accuracy for DecisionTree and Random Forest is:',ensemble.score(X_test, y_test))
    

def svm_clinical():
    df = pd.read_csv('./data/df_onsite.csv')
    lvo = df['lvo']
    features = df.drop(['lvo'], axis=1)
    path = r'models\svm_model.pkl'
    model, X_train, X_test, y_train, y_test = train_svm_clinical(features, lvo)
    return model, X_train, X_test, y_train, y_test


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

    all_features = pd.concat([eeg_features, clinical_features], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(all_features, lvo, test_size=0.2, random_state=42)

    X_train_eeg = X_train[X_train.columns[0:8]]
    X_train_clinical = X_train[X_train.columns[8:]]
    X_test_eeg = X_test[X_test.columns[0:8]]
    X_test_clinical = X_test[X_test.columns[8:]]
    
    #path = r'models\svm_model.pkl'
    eeg_model = train_svm_eeg(X_train_eeg, X_test_eeg, y_train, y_test)
    clinical_model = train_svm_clinical(X_train_clinical, X_test_clinical, y_train, y_test)

    combine(eeg_model, clinical_model, X_train, y_train, X_test, y_test)


    #return model, X_train, X_test, y_train, y_test

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
    # enter path here
    #Xtrain1 , X_test1, y_train1, y_test1, model1 = svm_clinical()
    svm_eeg()

    
    