import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import csv
import shap
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import CustomTrainDataset, CustomTestDataset
from torch.utils.data import Dataset, DataLoader

from models.lstm import LSTM1

from utils.plot_acc import plot_acc_loss
from sklearn.metrics import confusion_matrix


def main(lr, epoch, batch_size, num_layer):

    # Save the trained model
    PATH = './pretrained/lstm_trained.pth'

    # Load eeg data
    store = np.load('./data/processed_eeg.npy')

    # Load clinical data
    df = pd.read_csv('./data/df_onsite.csv')
    lvo = df['lvo'].to_numpy()
    # lvo = np.delete(lvo, 87)
    lvo = lvo.reshape(lvo.shape[0], -1)


    clinical_train, clinical_test, label_train, label_test = train_test_split(
        store, lvo, test_size=0.2, random_state=42)

    # Duplicate the data if num_layer > 1
    # if num_layer > 1:
    #     # store = np.repeat(store, num_layer, axis=1)
    #     label_train = np.repeat(label_train, num_layer, axis=0)
    #     label_test = np.repeat(label_test, num_layer, axis=0)

    train = CustomTrainDataset(torch.FloatTensor(
        clinical_train), torch.FloatTensor(label_train))
    test = CustomTrainDataset(torch.FloatTensor(
        clinical_test), torch.FloatTensor(label_test))

    

    # Define hyperparameters

    num_epochs = epoch  # 1000 epochs
    learning_rate = lr  # 0.001 lr

    input_size = store.shape[2]  # number of features
    hidden_size = 2  # number of features in hidden state
    num_layers = num_layer  # number of stacked lstm layers
    num_classes = 1  # number of output classes
    num_time = store.shape[1]

    lstm = LSTM1(num_classes, input_size, hidden_size,
                 num_layers, num_time)  # our lstm class

    # Train on GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    lstm.to(device)

    # Create a loss function and optimizer
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model on the training data
    lstm.train()
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test, shuffle=False)
        # X_train_tensors = X_train_tensors.to(device)
        for (idx, batch) in enumerate(trainloader):
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0
            outputs = lstm(inputs)  # forward pass

            # obtain the loss function
            if num_layer > 1:
                # store = np.repeat(store, num_layer, axis=1)
                labels = labels.repeat(num_layer,1)


            loss = criterion(outputs, labels)
            acc = binary_acc(outputs, labels)

            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop

            train_loss += loss.item()
            train_acc += acc.item()

        # lstm.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for (idx, data) in enumerate(testloader):
                test_inputs, test_labels = data[0], data[1]
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)

                test_outputs = lstm(test_inputs)

                if num_layer > 1:
                    test_labels = test_labels.repeat(num_layer, 1)

                loss = criterion(test_outputs, test_labels)
                acc = binary_acc(test_outputs, test_labels)

                test_loss += loss.item()
                test_acc += acc.item()

        print('Epoch {}: | Train Acc: {} | Test Acc: {}'.format(
            epoch, train_acc/len(trainloader), test_acc/len(testloader)))
        train_accs.append(train_acc/len(trainloader))
        train_losses.append(train_loss/len(trainloader))
        test_accs.append(test_acc/len(testloader))
        test_losses.append(test_loss/len(testloader))
    # Save model
    torch.save(lstm.state_dict(), PATH)
    
    # Plot the 
    # Test the model 
    label_pred_list = []
    lstm.eval()
    with torch.no_grad():
        for obj in testloader:
            inputs = obj[0].to(device)
            labels = obj[1].to(device)
            label_test_pred = lstm(inputs)
            label_pred_tag = torch.round(label_test_pred)
            label_pred_list.append(label_pred_tag.cpu().numpy())

    label_pred_list = [a.squeeze().tolist() for a in label_pred_list]
    print('Confusion matrix')
    CM = confusion_matrix(label_test, label_pred_list)
    print(CM)
    print(evaluation_metric(CM))

    # Plot the accuracy and loss
    plot_acc_loss(train_accs, test_accs, train_losses,
                  test_losses, "Acc and Loss")
    plt.show()

    # Save the result to the csv file
    avg_train_accs = sum(train_accs)/len(train_accs)
    avg_train_losses = sum(train_losses)/len(train_losses)
    avg_test_accs = sum(test_accs)/len(test_accs)
    avg_test_losses = sum(test_losses)/len(test_losses)
    print("Train acc: {} | Train loss: {} | Test acc: {} | Test loss: {}".format(avg_train_accs,avg_train_losses, avg_test_accs,avg_test_losses))

    

    with open('./results/result-lstm.csv','w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([avg_train_accs, avg_train_losses, avg_test_accs, avg_test_losses])
        

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def evaluation_metric(CM):
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    expected_loss = (4*FN+FP)/(4*TP+TN)
    return expected_loss

def shap_values(X_train, X_test, model, features):
    # Use the training data for deep explainer => can use fewer instances
    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    # init the JS visualization code
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--lr', type=float, required=False,
                        default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False,
                        default=1, help='Number of epoch')
    parser.add_argument('--batch_size', type=int,
                        required=False, default=4, help='Size of batch')
    parser.add_argument('--num_layers', type=int, required=False,
                        default=2, help='Number of stacked lstm layers')

    args = parser.parse_args()
    lr = args.lr
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    num_layers = args.num_layers

    main(lr, num_epoch, batch_size, num_layers)
