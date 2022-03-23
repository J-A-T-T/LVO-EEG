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

def train_svm_classifer(model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # Load the datasets

    lvo = pd.read_csv('./data/df_onsite.csv')
    lvo = lvo['lvo']
    eeg_features = pd.read_csv(r'data\feature_processed\simple_features.csv')

    # Normalize the data using z-score
    scaler = StandardScaler()
    scaler.fit(eeg_features)
    transformer = FunctionTransformer(zscore)
    eeg_features = transformer.transform(eeg_features)

    # Split into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(eeg_features, lvo, test_size=0.2, random_state=42)

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

    # 6-fold cross validation
    clf = GridSearchCV(svm, param,
            cv=6, verbose=3)

    clf.fit(X_train, y_train)

    if os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict=clf.predict(X_test)

    print("Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_test, y_predict)))

    print("\nLoss Metric:")
    cm = confusion_matrix(y_test, y_predict)
    eval = evaluation_metric(cm)
    print(eval)

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

def evaluation_metric(CM):
    TP = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TN = CM[1][1]

    expected_loss = (4*FN+FP)/(4*TP+TN)
    return expected_loss
 
if __name__ == '__main__':
    # enter path here
    path = r'results'
    train_svm_classifer(path)