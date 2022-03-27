from .NN import NeuralNet
from .eegnet import EEGNet

import torch.nn as nn
import torch.nn.functional as F
import torch

class LVOEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.clinical = NeuralNet(num_features_in = 5, num_features_out= 8, embed_dim = 128)
        self.eegnet = EEGNet(output=8)
        # Activation and regularization
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer1 = nn.Linear(16,8)
        self.layer2 = nn.Linear(8,1)

    def forward(self, clinical, eeg):
        clinical = self.clinical(clinical)
        eeg = self.eegnet(eeg)

        x = torch.cat((clinical,eeg),1)
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
    