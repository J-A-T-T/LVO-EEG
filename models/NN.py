import torch.nn as nn
import torch.nn.functional as F
import torch

class NeuralNet(nn.Module):
    def __init__(self, num_features_in = 5, num_features_out= 1, embed_dim = 128):
        super().__init__()
        self.layer1 = nn.Linear(num_features_in, embed_dim)
        nn.init.xavier_uniform_(tensor=self.layer1.weight)
        
        self.layer2 = nn.Linear(embed_dim, embed_dim*2)
        nn.init.xavier_uniform_(tensor=self.layer2.weight)
        
        self.layer3 = nn.Linear(embed_dim*2, embed_dim*2)
        nn.init.xavier_uniform_(tensor=self.layer3.weight)
        
        self.layer4 = nn.Linear(embed_dim*2, num_features_out)
        nn.init.xavier_uniform_(tensor=self.layer4.weight)

        # Activation and regularization
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batchnorm1 = nn.BatchNorm1d(embed_dim)
        self.batchnorm2 = nn.BatchNorm1d(embed_dim*2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        # x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        
        
        x = self.layer4(x)
        x = self.sigmoid(x)
        return x
    