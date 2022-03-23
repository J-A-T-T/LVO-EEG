import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer


def train_test_data():
    # Load the label 
    lvo = pd.read_csv('./data/df_onsite.csv')
    lvo = lvo['lvo']
    eeg_features = pd.read_csv(r'C:\Users\akars\Desktop\Studies\UNI\CMPUT 469\Project\LVO-EEG\data\feature_processed\simple_features.csv')

    print(eeg_features.columns)
    # Split into training set and test set
    scaler = StandardScaler()
    scaler.fit(eeg_features)
    transformer = FunctionTransformer(zscore)
    eeg_features = transformer.transform(eeg_features)

    x_train, x_test, y_train,  y_test = train_test_split(eeg_features, lvo, test_size=0.3)
    
    return x_train, x_test, y_train, y_test


def model(X_train, X_test, y_train, y_test):
    linear_svc = svm.SVC(kernel='linear', C=1.0, probability=True).fit(X_train, y_train)
    rbf_svc = svm.SVC(kernel='rbf', C=1.0, probability=True).fit(X_train, y_train)
    poly_svc = svm.SVC(kernel='poly', C=1.0, probability=True).fit(X_train, y_train)
    sigmoid_svc = svm.SVC(kernel='sigmoid', C=1.0, probability=True).fit(X_train, y_train)

    return linear_svc, rbf_svc, poly_svc, sigmoid_svc


def preds(linear_svc, rbf_svc, poly_svc, sigmoid_svc, X_test):
    linear_svc_pred = linear_svc.predict(X_test)
    rbf_svc_pred = rbf_svc.predict(X_test)
    poly_svc_pred = poly_svc.predict(X_test)
    sigmoid_svc_pred = sigmoid_svc.predict(X_test)

    return linear_svc_pred, rbf_svc_pred, poly_svc_pred, sigmoid_svc_pred

def evaluation_metric(CM):
    TP = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TN = CM[1][1]

    expected_loss = (4*FN+FP)/(4*TP+TN)
    return expected_loss


if __name__ == '__main__':


    X_train, X_test, y_train, y_test = train_test_data()
    linear_svc, rbf_svc, poly_svc, sigmoid_svc = model(X_train, X_test, y_train, y_test)
    linear_svc_pred, rbf_svc_pred, poly_svc_pred, sigmoid_svc_pred = preds(linear_svc, rbf_svc, poly_svc, sigmoid_svc, X_test)

    linear_svc_CM  = confusion_matrix(y_test,linear_svc_pred )
    rbf_svc_CM  = confusion_matrix(y_test,rbf_svc_pred )
    poly_svc_CM = confusion_matrix(y_test,poly_svc_pred)
    sigmoid_svc_CM  = confusion_matrix(y_test,sigmoid_svc_pred )

    linear_svc_F1  = accuracy_score(y_test,linear_svc_pred )
    rbf_svc_F1  = accuracy_score(y_test,rbf_svc_pred )
    poly_svc_F1 = accuracy_score(y_test,poly_svc_pred)
    sigmoid_svc_F1  = accuracy_score(y_test,sigmoid_svc_pred )

    linear_svc_loss  = evaluation_metric(linear_svc_CM)
    rbf_svc_loss  = evaluation_metric(rbf_svc_CM)
    poly_svc_loss = evaluation_metric(poly_svc_CM)
    sigmoid_svc_loss  = evaluation_metric(sigmoid_svc_CM)

    print(linear_svc_F1)
    print(rbf_svc_F1)
    print(poly_svc_F1)
    print(sigmoid_svc_F1)

    print(linear_svc_loss)
    print(rbf_svc_loss)
    print(poly_svc_loss)
    print(sigmoid_svc_loss)
    #print(linear_svc_pred.shape)
    #print('Training set score: {:.4f}'.format(accuracy_score(X_train, y_train)))