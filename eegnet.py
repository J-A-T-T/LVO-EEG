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
from utils.utils import CustomTrainDataset, CustomTestDataset 
from torch.utils.data import Dataset, DataLoader
from utils.plot_acc import plot_acc_loss

from models.eegnet import EEGNet

def main(lr, epochs, batch_size):

    # Save the trained model
    PATH = './pretrained/eegnet_trained.pth'

    # Load the label 
    df = pd.read_csv('./data/df_onsite.csv')
    lvo = df['lvo'].to_numpy()
    lvo = np.delete(lvo, 87)
    lvo = np.float32(lvo)
    
    # Load the preprocessed eeg data
    store = np.load('./data/processed_eeg.npy')
    store = store.reshape(store.shape[0], 1, store.shape[1], store.shape[2])
    store = np.float32(store)

    # Split into training set and test set

    x_train, x_test, y_train,  y_test = train_test_split(store, lvo, test_size=0.2, random_state=42)

    train = CustomTrainDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    test = CustomTrainDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test, shuffle=False)

    # Set up the model and optimizer
    net = EEGNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)


    test_results=[]
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0
        train_acc = 0
        running_loss = 0.0
        for (idx, batch) in enumerate(trainloader):
            # s = i*batch_size
            # e = i*batch_size+batch_size
            # inputs = torch.from_numpy(x_train[s:e])
            # labels = torch.FloatTensor(np.array([y_train[s:e]]).T*1.0)
            inputs, labels = batch[0], batch[1]
            # wrap them in Variable
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            acc = binary_acc(outputs, labels.unsqueeze(1))

            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            train_loss += loss.item()
            train_acc += acc.item()
        
        # Validation accuracy
        net.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for (idx, data) in enumerate(testloader):
                test_inputs, test_labels = data[0], data[1]
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                
                test_outputs = net(test_inputs)
                
                loss = criterion(test_outputs, test_labels.unsqueeze(1))
                acc = binary_acc(test_outputs, test_labels.unsqueeze(1))
                
                test_loss += loss.item()
                test_acc += acc.item()
        print('Epoch {}: | Train Acc: {} | Test Acc: {}'.format(epoch, train_acc/len(trainloader), test_acc/len(testloader)))
        train_accs.append(train_acc/len(trainloader))
        train_losses.append(train_loss/len(trainloader))
        test_accs.append(test_acc/len(testloader))
        test_losses.append(test_loss/len(testloader))
    
    # Save model
    torch.save(net.state_dict(), PATH)

        # Plot the accuracy and loss
    plot_acc_loss(train_accs, test_accs, train_losses, test_losses, "Acc and Loss")



# def evaluate(model, X, Y, params = ["acc"]):
#     results = []
#     batch_size = 4
    
#     predicted = []
    
#     for i in range(int(len(X)/batch_size)):
#         s = i*batch_size
#         e = i*batch_size+batch_size
        
#         inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
#         pred = model(inputs)
        
#         predicted.append(pred.data.cpu().numpy())
        
        
#     inputs = Variable(torch.from_numpy(X).cuda(0))
#     predicted = model(inputs)
    
#     predicted = predicted.data.cpu().numpy()
    
#     for param in params:
#         if param == 'acc':
#             results.append(accuracy_score(Y, np.round(predicted)))

#     return results  

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=100, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='Size of batch')

    args = parser.parse_args()
    lr = args.lr
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    main(lr, num_epoch, batch_size)