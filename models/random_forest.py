import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


def train_rf_clinical_and_eeg(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators= 1, min_samples_split =2, min_samples_leaf= 4, max_features= 'auto', max_depth= 1, bootstrap= True).fit(X_train, y_train)

    print("\nRandom Forest:")
    print("Train Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_train, rf.predict(X_train))))
    print("Test Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_test, rf.predict(X_test))))
    print("Train Custom Evaluation Metric: {0}".format(evaluation_metric(y_train, rf.predict(X_train))))
    print("Test Custom Evaluation Metric: {0}".format(evaluation_metric(y_test, rf.predict(X_test))))

    return rf


def train_rf_clinical(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(n_estimators= 1555, min_samples_split =2, min_samples_leaf= 1, max_features= 'auto', max_depth= 1, bootstrap= True).fit(X_train, y_train)

    print("\nRandom Forest:")
    print("Train Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_train, rf.predict(X_train))))
    print("Test Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_test, rf.predict(X_test))))
    print("Train Custom Evaluation Metric: {0}".format(evaluation_metric(y_train, rf.predict(X_train))))
    print("Test Custom Evaluation Metric: {0}".format(evaluation_metric(y_test, rf.predict(X_test))))


    return rf
    
def rf_clinical():
    df = pd.read_csv(r"..\data\df_onsite.csv")
    df['onset_d_t'] = pd.to_datetime(df['onset_d_t'])
    df['eeg_d_t'] = pd.to_datetime(df['eeg_d_t'])
    df['time_elapsed'] = df['eeg_d_t'] - df['onset_d_t']
    df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'])
    df['time_elapsed'] = df['time_elapsed'].dt.total_seconds().div(60).astype(int)
    onsite_features = ['age', 'gender', 'lams', 'lvo','time_elapsed']
    df = df[onsite_features]
    lvo = df['lvo']
    features = df.drop(['lvo'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, lvo, test_size=0.2, random_state=42)

    model = train_rf_clinical(X_train, X_test, y_train, y_test)
    if run_HyperParameterTuning:
        x = randomHyperParameterTuning(X_train, y_train)
        print(x)
    return model


def rf_clinical_and_eeg():
    df = pd.read_csv(r"..\data\df_onsite.csv")
    df['onset_d_t'] = pd.to_datetime(df['onset_d_t'])
    df['eeg_d_t'] = pd.to_datetime(df['eeg_d_t'])
    df['time_elapsed'] = df['eeg_d_t'] - df['onset_d_t']
    df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'])
    df['time_elapsed'] = df['time_elapsed'].dt.total_seconds().div(60).astype(int)
    onsite_features = ['age', 'gender', 'lams', 'lvo','time_elapsed']
    df = df[onsite_features]
    lvo = df['lvo']
    clinical_features = df.drop(['lvo'], axis=1)
    eeg_features = pd.read_csv(r'..\data\feature_processed\simple_features.csv')
    all_features = pd.concat([eeg_features, clinical_features], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(all_features, lvo, test_size=0.2, random_state=42)
    model = train_rf_clinical_and_eeg(X_train, X_test, y_train, y_test)
    if run_HyperParameterTuning_eeg:
        x = randomHyperParameterTuning(X_train, y_train)
        print(x)
    return model


def model(X_train, y_train):
    rand_attempt1 = RandomForestClassifier(n_estimators= 1, min_samples_split =2, min_samples_leaf= 4, max_features= 'auto', max_depth= 1, bootstrap= True).fit(X_train, y_train)
    return rand_attempt1


def preds(forest_classifier, X_test):
    forest_pred = forest_classifier.predict(X_test)

    return forest_pred

def randomHyperParameterTuning(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(start = 1, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [1, 2, 14]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    rf =RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    return rf_random.best_params_


def evaluation_metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    expected_loss = (4*FN+FP)/(4*(TP+FP)+(TN+FN))
    return expected_loss


def acc(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)

if __name__ == '__main__':
    run_HyperParameterTuning = False
    if input("Run HyperParameterTuning? (y/n)") == 'y':
        run_HyperParameterTuning = True
    run_HyperParameterTuning_eeg = False
    if input("Run HyperParameterTuning_eeg? (y/n)") == 'y':
        run_HyperParameterTuning_eeg = True
    
    
    clinical_model = rf_clinical()
    clinical_eeg_model = rf_clinical_and_eeg()

    if run_HyperParameterTuning:
        print("coolies")

