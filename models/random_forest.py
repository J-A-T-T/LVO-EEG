import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


def train_test_data():
    # Load the label 
    lvo = pd.read_csv(r"C:\Users\tanya\OneDrive\Documents\GitHub\LVO-EEG\data\df_onsite.csv")
    lvo = lvo['lvo']
    eeg_features = pd.read_csv(r'C:\Users\tanya\OneDrive\Documents\GitHub\LVO-EEG\data\feature_processed\simple_features.csv')

    print(eeg_features.columns)
    # Scale data
    scaler = StandardScaler()
    scaler.fit(eeg_features)

    # transformer = FunctionTransformer(zscore)
    eeg_features = scaler.transform(eeg_features)
    # Split into training set and test set
    x_train, x_test, y_train,  y_test = train_test_split(eeg_features, lvo, test_size=0.3)
    
    return x_train, x_test, y_train, y_test


def model(X_train, X_test, y_train, y_test):
    rand_attempt1 = RandomForestClassifier(n_estimators= 1557, min_samples_split =2, min_samples_leaf= 4, max_features= 'sqrt', max_depth= 77, bootstrap= True).fit(X_train, y_train)
    return rand_attempt1


def preds(forest_classifier, X_test):
    forest_pred = forest_classifier.predict(X_test)

    return forest_pred

def randomHyperParameterTuning(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(1, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
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


def evaluation_metric(CM):
    TP = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TN = CM[1][1]

    expected_loss = (4*FN+FP)/(4*TP+TN)
    return expected_loss


if __name__ == '__main__':
    run_HyperParameterTuning = False
    if input("Run HyperParameterTuning? (y/n)") == 'y':
        run_HyperParameterTuning = True
    
    
    X_train, X_test, y_train, y_test = train_test_data()
    forest_attempt = model(X_train, X_test, y_train, y_test)
    forest_pred = preds(forest_attempt, X_test)

    forest_CM  = confusion_matrix(y_test,forest_pred )

    forest_F1  = accuracy_score(y_test,forest_pred )

    forest_loss  = evaluation_metric(forest_CM)
    print(forest_F1)

    print(forest_loss)
    
    if run_HyperParameterTuning:
        print(randomHyperParameterTuning(X_train, y_train))