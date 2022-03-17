# Import libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# create custom dataset class
class CustomTrainDataset(Dataset):
    def __init__(self, clinical, labels):
        self.labels = labels
        self.clinical = clinical

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.clinical[idx], self.labels[idx]

class CustomTestDataset(Dataset):
    def __init__(self, clinical):
        self.clinical = clinical

    def __len__(self):
        return len(self.clinical)

    def __getitem__(self, idx):
        return self.clinical[idx]