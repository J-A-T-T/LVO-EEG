from .NN import NeuralNet
from .lstm import LSTM1

import torch.nn as nn
import torch.nn.functional as F
import torch

class LVONet(nn.Module):
    def __init__(self, num_layer):
        super().__init__()
        self.clinical = NeuralNet(num_features_in = 5, num_features_out= 8, embed_dim = 128)
        self.lstm = LSTM1(num_classes=8, input_size=4, hidden_size=2,num_layers=num_layer, seq_length=5000)
        self.num_layer = num_layer
        # Activation and regularization
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer1 = nn.Linear(16,8)
        self.layer2 = nn.Linear(8,1)

    def forward(self, clinical, eeg):
        clinical = self.clinical(clinical)
        eeg = self.lstm(eeg)

        if self.num_layer > 1:
            # store = np.repeat(store, num_layer, axis=1)
            clinical = clinical.repeat(self.num_layer,1)
        

        x = torch.cat((clinical,eeg),1)
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
    