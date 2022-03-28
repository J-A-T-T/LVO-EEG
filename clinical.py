import argparse
import csv
from audioop import avg
from math import e
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import eli5
# from eli5.sklearn import PermutationImportance

import shap

import torch
from utils.utils import CustomTrainDataset, CustomTestDataset 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim

from models.NN import NeuralNet

from utils.plot_acc import plot_acc
from utils.plot_loss import plot_loss
from utils.plot_custom_eval import plot_custom_eval
# from utils.plot_acc import plot_acc_loss

def main(lr, num_epoch, batch_size):

    # Define hyperparameters
    epochs = num_epoch
    k_folds = 6
    torch.manual_seed(42) #Tuning this parameter

    # Save the trained model
    PATH = './pretrained/clinical_trained.pth'

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
    # test = CustomTestDataset(torch.FloatTensor(clinical_test.values))
    test = CustomTrainDataset(torch.FloatTensor(clinical_test.values), torch.FloatTensor(label_test.values))
    dataset = ConcatDataset([train, test])
    # # Create DataLoader
    # trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    # testloader = DataLoader(test, shuffle=False)

    # Create k-fold cross validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    result = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('--------')
        print('Fold {}'.format(fold))
        print('--------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch_size, sampler=test_subsampler)
        

        # Create a model
        model = NeuralNet()

        # Training on GPU if possible
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # print(device)
        model.to(device)

        # Create a Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # Train the model on training data
        model.train() 
        train_accs = []
        test_accs = []
        train_losses = []
        test_losses = []
        train_custom_evals = []
        test_custom_evals = []
        for epoch in range(1, epochs+1):
            train_loss = 0
            train_acc = 0
            label_train_list = []
            label_train_epoch = []
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

                label_train_tag = torch.round(outputs)
                label_train_list.append(label_train_tag.detach().cpu().numpy())
                label_train_epoch.append(labels.detach().cpu().numpy())

                train_loss += loss.item()
                train_acc += acc.item()
            
            label_train_list = [a.squeeze().tolist() for a in label_train_list]
            label_train_result = sum(label_train_list, [])
            label_train_epoch  = [a.squeeze().tolist() for a in label_train_epoch]
            label_train_epoch = sum(label_train_epoch, [])
            CM_train = confusion_matrix(label_train_epoch, label_train_result)
            custom_train_evaluation = evaluation_metric(CM_train)

            # Test the model 
            label_test_list = []
            label_test_epoch = []
            model.eval()
            test_loss = 0
            test_acc = 0
            with torch.no_grad():
                for (idx, data) in enumerate(testloader):
                    test_inputs, test_labels = data[0], data[1]
                    test_inputs = test_inputs.to(device)
                    test_labels = test_labels.to(device)
                    
                    test_outputs = model(test_inputs)
                    
                    loss = criterion(test_outputs, test_labels.unsqueeze(1))
                    acc = binary_acc(test_outputs, test_labels.unsqueeze(1))
                    
                    label_test_tag = torch.round(test_outputs)
                    label_test_list.append(label_test_tag.cpu().numpy())
                    label_test_epoch.append(test_labels.cpu().numpy())

                    test_loss += loss.item()
                    test_acc += acc.item()
                
                label_test_list = [a.squeeze().tolist() for a in label_test_list]
                label_test_epoch = [a.squeeze().tolist() for a in label_test_epoch]
                label_test_list = sum(label_test_list,[])
                label_test_epoch = sum(label_test_epoch,[])
                CM_test = confusion_matrix(label_test_epoch, label_test_list)
                custom_test_evaluation = evaluation_metric(CM_test)

            scheduler.step(test_loss)
            print('Epoch {}: | Train Loss: {:.4f} | Train Acc: {:.4f} | Train Custom Eval: {:.4f} | Test Loss: {:.4f} | Test Acc: {:.4f} | Test Custom Eval: {:.4f}'.format(epoch, train_loss/len(trainloader), train_acc/len(trainloader), custom_train_evaluation/len(trainloader),test_loss/len(testloader), test_acc/len(testloader), custom_test_evaluation/len(testloader)))
            train_accs.append(train_acc/len(trainloader))
            train_losses.append(train_loss/len(trainloader))
            test_accs.append(test_acc/len(testloader))
            test_losses.append(test_loss/len(testloader))
            train_custom_evals.append(custom_train_evaluation/len(trainloader))
            test_custom_evals.append(custom_test_evaluation/len(testloader))

        avg_train_accs = sum(train_accs)/len(train_accs)
        avg_train_losses = sum(train_losses)/len(train_losses)
        avg_train_custom_eval = sum(train_custom_evals)/len(train_custom_evals)
        avg_test_accs = sum(test_accs)/len(test_accs)
        avg_test_losses = sum(test_losses)/len(test_losses)
        avg_test_custom_eval = sum(test_custom_evals)/len(test_custom_evals)
        print("Fold: {} | Test loss: {:.4f} | Test acc: {:.4f} | Test custom eval: {:.4f} ".format(fold, avg_test_losses, avg_test_accs, avg_test_custom_eval))

        result[fold] = {"avg_test_accs": avg_test_accs, "avg_test_losses": avg_test_losses, "avg_test_custom_eval": avg_test_custom_eval, "train_accs": train_accs, "train_losses":train_losses, "train_custom_eval": train_custom_evals, "test_accs": test_accs, "test_losses": test_losses, "test_custom_eval": test_custom_evals}

    avg_loss = (result[0]["avg_test_losses"] + result[1]["avg_test_losses"] + result[2]["avg_test_losses"] + result[3]["avg_test_losses"] + result[4]["avg_test_losses"] + result[5]["avg_test_losses"])/6
    avg_acc = (result[0]["avg_test_accs"] + result[1]["avg_test_accs"] + result[2]["avg_test_accs"] + result[3]["avg_test_accs"] + result[4]["avg_test_accs"] + result[5]["avg_test_accs"])/6
    avg_custom_eval = (result[0]["avg_test_custom_eval"] + result[1]["avg_test_custom_eval"] + result[2]["avg_test_custom_eval"] + result[3]["avg_test_custom_eval"] + result[4]["avg_test_custom_eval"] + result[5]["avg_test_custom_eval"])/6

    train_loss = (np.array(result[0]["train_losses"]) + np.array(result[1]["train_losses"]) + np.array(result[2]["train_losses"]) + np.array(result[3]["train_losses"]) + np.array(result[4]["train_losses"]) + np.array(result[5]["train_losses"]))/6
    test_loss = (np.array(result[0]["test_losses"]) + np.array(result[1]["test_losses"]) + np.array(result[2]["test_losses"]) + np.array(result[3]["test_losses"]) + np.array(result[4]["test_losses"]) + np.array(result[5]["test_losses"]))/6
    train_acc = (np.array(result[0]["train_accs"]) + np.array(result[1]["train_accs"]) + np.array(result[2]["train_accs"]) + np.array(result[3]["train_accs"]) + np.array(result[4]["train_accs"]) + np.array(result[5]["train_accs"]))/6
    test_acc = (np.array(result[0]["test_accs"]) + np.array(result[1]["test_accs"]) + np.array(result[2]["test_accs"]) + np.array(result[3]["test_accs"]) + np.array(result[4]["test_accs"]) + np.array(result[5]["test_accs"]))/6
    train_custom_eval  = (np.array(result[0]["train_custom_eval"]) + np.array(result[1]["train_custom_eval"]) + np.array(result[2]["train_custom_eval"]) + np.array(result[3]["train_custom_eval"]) + np.array(result[4]["train_custom_eval"]) + np.array(result[5]["train_custom_eval"]))/6
    test_custom_eval = (np.array(result[0]["test_custom_eval"]) + np.array(result[1]["test_custom_eval"]) + np.array(result[2]["test_custom_eval"]) + np.array(result[3]["test_custom_eval"]) + np.array(result[4]["test_custom_eval"]) + np.array(result[5]["test_custom_eval"]))/6
    
    print("The average performance of {}-fold CV".format(fold))
    print("The average loss of {}-fold CV: {:.4f}".format(fold, avg_loss))
    print("The average accuracy of {}-fold CV: {:.4f} ".format(fold, avg_acc))
    print("The average custom evaluation of {}-fold CV: {:.4f}".format(fold, avg_custom_eval))

    # Save model
    torch.save(model.state_dict(), PATH)

    # Plot the accuracy, loss, and 
    plot_loss(train_loss, test_loss, "Loss")
    plot_acc(train_acc, test_acc, "Acc")
    plot_custom_eval(train_custom_eval, test_custom_eval, "Custom Evaluation")

    with open('./results/result-clinical.csv','w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([avg_loss, avg_acc, avg_custom_eval])


    # Provide SHAP values: Try other explainer than GradientExplainer
    # SHAP values represent a features's responsibility for a change in the model output 
    e = shap.GradientExplainer(model, torch.FloatTensor(clinical_train.values).to(device))
    shap_values = e.shap_values(torch.FloatTensor(clinical_test.values).to(device))
    # print(shap_values)
    x_test_values = clinical_test.to_numpy()
    shap.summary_plot(shap_values, x_test_values, feature_names=features)

    # Provide Grad-Cam

    
    
def custom_lost_function(outputs, labels, device):
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
    modCriterion = nn.BCELoss(weight=weightArr.to(device))
    return modCriterion(outputs, labels)


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

    expected_loss = (4*FN+FP)/(4*(TP+FP)+TN+FN)
    return expected_loss
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--lr', type=float, required=False, default=1e-5, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=100, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='Size of batch')

    args = parser.parse_args()
    lr = args.lr
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    main(lr, num_epoch, batch_size)
