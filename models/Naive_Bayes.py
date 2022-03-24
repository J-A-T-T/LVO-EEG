from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def train():
    """
    train naive bayes model
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

    #Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(X_train, y_train)

    y_predict_train = model.predict(X_train)
    y_predict=model.predict(X_test)

    print("Train Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_train, y_predict_train)))
    print("Test Accuracy: {0}".format(sklearn.metrics.accuracy_score(y_test, y_predict)))

    print("\nLoss Metric:")
    cm = confusion_matrix(y_test, y_predict)
    cm_train = confusion_matrix(y_train, y_predict_train)
    eval = evaluation_metric(cm)  
    eval_train = evaluation_metric(cm_train)

    print("Train Loss : {0}".format(eval_train))
    print("Test Loss : {0}".format(eval))

def evaluation_metric(CM):
    TP = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TN = CM[1][1]

    expected_loss = (4*FN+FP)/(4*TP+TN)
    return expected_loss
 
if __name__ == '__main__':

    train()