from sklearn.utils import shuffle
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import CustomClinicalDataset 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.conv import ConvNet

def main():

    # Define hyperparameters
    batch_size = 4 
    epochs = 100

    # Save the trained model
    PATH = './pretrained/trained.pth'

    #Load clinical data first
    df = pd.read_csv('./data/df_onsite.csv')

    # Separate data into training and testing data

    clinical = df[['age', 'lams', 'nihss', 'time_elapsed', 'Male', 'Female']].to_numpy()
    label = df['lvo'].to_numpy()

    clinical_train, clinical_test, label_train, label_test = train_test_split(clinical, label, test_size = 0.2, random_state=42)

    # Define the custom dataset
    train = CustomClinicalDataset(clinical_train, label_train)
    test = CustomClinicalDataset(clinical_test, label_test)
    # print('First iteration of dataset: ', next(iter(TD)), '\n')

    # Create DataLoader
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test, shuffle=False)

    # Creat a model
    net = ConvNet()

    # Training on GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)

    # Create a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the model on training data
    for epoch in range(epochs):
        running_loss = 0

        for (idx, batch) in enumerate(trainloader):
            inputs, labels = batch['Clinical'], batch['Class']

            # Cast the data to be in correct format
            inputs = inputs.float()
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx %2 == 1:
                print(running_loss/2)
                running_loss = 0

    print("Finished training")
    torch.save(net.state_dict(), PATH)
    
    # Test the model on the test data
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['Clinical'], data['Class']

            inputs = inputs.float()
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network: {100 * correct // total} %')
    

main()