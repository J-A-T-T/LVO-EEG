import argparse
from audioop import avg
from math import e
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import shap

shap.

import torch
from utils import CustomTrainDataset, CustomTestDataset 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.NN import NeuralNet

def main(lr, num_epoch, batch_size):

    # Define hyperparameters
    epochs = num_epoch

    # Save the trained model
    PATH = './pretrained/trained.pth'

    #Load clinical data first
    df = pd.read_csv('./data/df_onsite.csv')
    result_file_name = 'result.csv'

    # Separate data into training, validation and testing data
    # clinical = df[['age', 'lams', 'nihss', 'time_elapsed', 'Male', 'Female']] 
    clinical = df[['age', 'lams', 'time_elapsed', 'Male', 'Female']] # Eliminate the nihss score
    features = clinical.columns
    label = df['lvo']
    clinical_train, clinical_test, label_train, label_test = train_test_split(clinical, label, test_size = 0.2, random_state=42)


    # Define the custom dataset

    train = CustomTrainDataset(torch.FloatTensor(clinical_train.values), torch.FloatTensor(label_train.values))
    test = CustomTestDataset(torch.FloatTensor(clinical_test.values))

    # Create DataLoader
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test, shuffle=False)

    # Creat a model
    model = NeuralNet()

    # Training on GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    # Create a Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model on training data
    model.train() 
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        
        for (idx, batch) in enumerate(trainloader):
            inputs, labels = batch[0], batch[1]

            # Cast the data to be in correct format
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            acc = binary_acc(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        print('Epoch {}: | Loss: {} | Acc: {}'.format(epoch, epoch_loss/len(trainloader), epoch_acc/len(trainloader)))

    print("Finished training")
    torch.save(model.state_dict(), PATH)

    # Test the model 

    label_pred_list = []
    model.eval()
    with torch.no_grad():
        for inputs in testloader:
            inputs = inputs.to(device)
            label_test_pred = model(inputs)
            label_pred_tag = torch.round(label_test_pred)
            label_pred_list.append(label_pred_tag.cpu().numpy())

    label_pred_list = [a.squeeze().tolist() for a in label_pred_list]

    # Classification report
    
    print('Confusion matrix')
    print(confusion_matrix(label_test, label_pred_list))

    print('Classification report')
    print(classification_report(label_test, label_pred_list))

    # Accuracy for test set: 85-87%
    print('Accuracy report')
    print(accuracy_score(label_test, label_pred_list))

    # Provide SHAP values
    # SHAP values represent a features's responsibility for a change in the model output 
    # e = shap.GradientExplainer(model, torch.FloatTensor(clinical_train.values).to(device))
    # shap_values = e.shap_values(torch.FloatTensor(clinical_test.values).to(device))
    # # print(shap_values)
    # x_test_values = clinical_test.to_numpy()
    # shap.summary_plot(shap_values, x_test_values, feature_names=features)
    # shap.force_plot(e.expected_value, shap_values)
    # ind = 0
    # shap.force_plot(
    #     e.expected_value, shap_values[ind,:], x_test_values[ind,:], features_name=features
    # )
    
    # shap.plots.force(shap_values)

    print(shap.__version__)
    # Provide Grad-Cam
    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=10, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='Size of batch')

    args = parser.parse_args()
    lr = args.lr
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    main(lr, num_epoch, batch_size)