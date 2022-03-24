import argparse
import csv
from audioop import avg
from math import e
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import shap
from sklearn.utils import shuffle

import torch
from utils.utils import CustomTrainDataset, CustomDataset 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.lvoeegnet import LVOEEGNet

from utils.plot_acc import plot_acc_loss

def main(lr, num_epoch, batch_size):

    # Define hyperparameters
    epochs = num_epoch

    # Save the trained model
    PATH = './pretrained/trained.pth'

    # Load the preprocessed eeg data
    eeg = np.load('./data/processed_eeg.npy')
    eeg = eeg.reshape(eeg.shape[0],1,eeg.shape[1], eeg.shape[2])
    
    #Load clinical data first
    df = pd.read_csv('./data/df_onsite.csv')
    result_file_name = 'result.csv'

    # Separate data into training, validation and testing data
    # clinical = df[['age', 'lams', 'nihss', 'time_elapsed', 'Male', 'Female']] 
    clinical = df[['age', 'lams', 'time_elapsed', 'Male', 'Female']] # Eliminate the nihss score
    features = clinical.columns
    label = df['lvo']

    label = label.to_numpy()
    clinical = clinical.to_numpy()

    # clinical = np.delete(clinical, 87,0)
    # label = np.delete(label, 87)
    label = label.reshape(label.shape[0],-1)
    clinical_train, clinical_test, label_train, label_test = train_test_split(clinical, label, test_size = 0.2, shuffle=False)
    eeg_train, eeg_test = train_test_split(eeg, test_size=0.2, shuffle=False)


    # Define the custom dataset
    train = CustomDataset(torch.FloatTensor(clinical_train), torch.FloatTensor(label_train), torch.FloatTensor(eeg_train))
    # test = CustomTestDataset(torch.FloatTensor(clinical_test.values))
    test = CustomDataset(torch.FloatTensor(clinical_test), torch.FloatTensor(label_test), torch.FloatTensor(eeg_test))

    # Create DataLoader
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test, shuffle=False)

    # Create a model
    model = LVOEEGNet()

    # Training on GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    # Create a Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Train the model on training data
    model.train() 
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    i = 1
    for epoch in range(1, epochs+1):
        train_loss = 0
        train_acc = 0
        trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        #testloader = DataLoader(test, shuffle=True)
        for (idx, batch) in enumerate(trainloader):
            inputs, labels, eegs = batch[0], batch[1], batch[2]

            # Cast the data to be in correct format
            inputs = inputs.to(device)
            labels = labels.to(device)
            eegs = eegs.to(device)
            
            #Zero the parameter gradients
            optimizer.zero_grad()


            # Forward + backward + optimize
            outputs = model(inputs, eegs)

            loss = criterion(outputs, labels)
            
            acc = binary_acc(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()


        # Test the model 
        label_pred_list = []
        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for (idx, data) in enumerate(testloader):
                test_inputs, test_labels, test_eegs = data[0], data[1], data[2]
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                test_eegs = test_eegs.to(device)

                
                test_outputs = model(test_inputs, test_eegs)
                
                #loss = criterion(test_outputs, test_labels)
                loss = custom_lost_function(test_outputs, test_labels)
                acc = binary_acc(test_outputs, test_labels)
                
                test_loss += loss.item()
                test_acc += acc.item()

        print('Epoch {}: | Train Acc: {} | Test Acc: {}'.format(epoch, train_acc/len(trainloader), test_acc/len(testloader)))
        train_accs.append(train_acc/len(trainloader))
        train_losses.append(train_loss/len(trainloader))
        test_accs.append(test_acc/len(testloader))
        test_losses.append(test_loss/len(testloader))
        
        if (i % 10 == 0):
            avg_train_accs = sum(train_accs)/len(train_accs)
            avg_train_losses = sum(train_losses)/len(train_losses)
            avg_test_accs = sum(test_accs)/len(test_accs)
            avg_test_losses = sum(test_losses)/len(test_losses)
            print("Train acc: {} | Train loss: {} | Test acc: {} | Test loss: {}".format(avg_train_accs,avg_train_losses, avg_test_accs,avg_test_losses))
        i += 1

    # Save model
    torch.save(model.state_dict(), PATH)

    # Plot the accuracy and loss
    plot_acc_loss(train_accs, test_accs, train_losses, test_losses, "Acc and Loss")

    # Save the result to the csv file
    avg_train_accs = sum(train_accs)/len(train_accs)
    avg_train_losses = sum(train_losses)/len(train_losses)
    avg_test_accs = sum(test_accs)/len(test_accs)
    avg_test_losses = sum(test_losses)/len(test_losses)
    print("Train acc: {} | Train loss: {} | Test acc: {} | Test loss: {}".format(avg_train_accs,avg_train_losses, avg_test_accs,avg_test_losses))



    with open('./results/result-eegnet-lvo.csv','w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([avg_train_accs, avg_train_losses, avg_test_accs, avg_test_losses])
        

    # Plot the 
    # Test the model 
    label_pred_list = []
    model.eval()
    with torch.no_grad():
        for obj in testloader:
            inputs = obj[0].to(device)
            labels = obj[1].to(device)
            eegs = obj[2].to(device)
                 
            label_test_pred = model(inputs, eegs)
            label_pred_tag = torch.round(label_test_pred)
            label_pred_list.append(label_pred_tag.cpu().numpy())

    label_pred_list = [a.squeeze().tolist() for a in label_pred_list]
    # Classification report
    
    print('Confusion matrix')
    CM = confusion_matrix(label_test, label_pred_list)
    print(CM)
    print(evaluation_metric(CM))

    # print('Classification report')
    # print(classification_report(label_test, label_pred_list))

    # # Accuracy for test set: 85-87%
    # print('Accuracy report')
    # print(accuracy_score(label_test, label_pred_list))

    # Provide SHAP values: Try other explainer than GradientExplainer
    # SHAP values represent a features's responsibility for a change in the model output 
    # e = shap.GradientExplainer(model, torch.FloatTensor(clinical_train).to(device))
    # shap_values = e.shap_values(torch.FloatTensor(clinical_test.values).to(device))
    # print(shap_values)
    # x_test_values = clinical_test.to_numpy()
    # shap.summary_plot(shap_values, x_test_values, feature_names=features)

    # Provide Grad-Cam

    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

# output is the predicted value
def custom_lost_function(outputs, labels):
    rounded_outputs = torch.round(outputs)

    # confusion_vector = rounded_output / label

    # true_positives = torch.sum(confusion_vector == 1)
    # false_positives = torch.sum(confusion_vector == float('inf'))
    # true_negatives = torch.sum(torch.isnan(confusion_vector))
    # false_negatives = torch.sum(confusion_vector == 0)

    # specificity = true_negatives / (true_negatives + false_positives + 1e-10)
    # recall = true_positives / (true_positives + false_negatives + 1e-10)
    # return_val = 1.0 - (0.1 * specificity + 0.9 * recall)

    # Initialize all weight to 1 at first
    weightArr = torch.ones(rounded_outputs.size()[0], 1)

    for i in range(rounded_outputs.size()[0]):
        # False negative when output = 0 but labels != output (i.e., label = 1, since label can only be in {0,1})
        if (rounded_outputs[i][0] == 0) and (rounded_outputs[i][0] != labels[i][0]):
            weightArr[i] = 4.0
    modCriterion = nn.BCELoss(weight=weightArr)
    return modCriterion(outputs, labels)

def evaluation_metric(CM):
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    expected_loss = (4*FN+FP)/(4*TP+TN)
    return expected_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=50, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='Size of batch')



    args = parser.parse_args()
    lr = args.lr
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    main(lr, num_epoch, batch_size)