import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvNet(nn.Module):
    def __init__(self, num_features_in = 6, num_features_out= 2, embed_dim = 3):
        super().__init__()
        self.layer1 = nn.Linear(num_features_in, embed_dim)
        self.layer2 = nn.Linear(embed_dim, num_features_out)

        # Activation and regularization
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x)
        return x
    