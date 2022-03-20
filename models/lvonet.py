from .NN import NeuralNet
from .lstm import LSTM1

import torch.nn as nn
import torch.nn.functional as F
import torch

class LVONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.clinical = NeuralNet(num_features_in = 5, num_features_out= 4, embed_dim = 64)
        self.lstm = LSTM1(num_classes=4, input_size=4, hidden_size=2,
                 num_layers=4, num_time=120)

        # Activation and regularization
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer1 = nn.Linear(8,4)
        self.layer2 = nn.Linear(4,1)

    def forward(self, clinical, eeg):
        clinical = self.clinical(clinical)
        eeg = self.lstm(eeg)
        x = torch.cat((clinical,eeg),0)
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
    