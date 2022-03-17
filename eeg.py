import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split 
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from models.eegnet import EEGNet

def main():

    # Load the label 
    lvo = pd.read_csv('./data/df_onsite.csv')
    lvo = lvo['lvo'].to_numpy()[:83]
    lvo = np.float32(lvo)
    
    # Load the preprocessed eeg data
    store = np.load('./data/processed_eeg.npy')

    store = store.reshape(store.shape[0], 1, store.shape[1], store.shape[2])
    store = np.float32(store)

    # Split into training set and test set

    x_train, x_test, y_train,  y_test = train_test_split(store, lvo, test_size=0.2, random_state=42)

    # Set up the model and optimizer
    net = EEGNet().cuda(0)
    # print(net.forward(Variable(torch.tensor(store).cuda(0))))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())

    batch_size = 4

    test_results=[]
    for epoch in range(500):  # loop over the dataset multiple times
        print("\nEpoch ", epoch)
        
        running_loss = 0.0
        for i in range(int(len(x_train)/batch_size)-1):
            s = i*batch_size
            e = i*batch_size+batch_size
            inputs = torch.from_numpy(x_train[s:e])
            labels = torch.FloatTensor(np.array([y_train[s:e]]).T*1.0)
            
            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation accuracy
        params = ["acc"]
        print(params)
        print("Training Loss ", running_loss)
        print("Train - ", evaluate(net, x_train, y_train, params))
        test_result = evaluate(net, x_test, y_test, params)
        test_results += test_result
        print("Test - ", test_result)

    print(sum(test_results)/len(test_results))


def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 4
    
    predicted = []
    
    for i in range(int(len(X)/batch_size)):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
        pred = model(inputs)
        
        predicted.append(pred.data.cpu().numpy())
        
        
    inputs = Variable(torch.from_numpy(X).cuda(0))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu().numpy()
    
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        # if param == "auc":
        #     results.append(roc_auc_score(Y, predicted))
        # if param == "recall":
        #     results.append(recall_score(Y, np.round(predicted)))
        # if param == "precision":
        #     results.append(precision_score(Y, np.round(predicted)))
        # if param == "fmeasure":
        #     precision = precision_score(Y, np.round(predicted))
        #     recall = recall_score(Y, np.round(predicted))
        #     results.append(2*precision*recall/ (precision+recall))
    return results  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=100, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='Size of batch')

    args = parser.parse_args()
    lr = args.lr
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    main()