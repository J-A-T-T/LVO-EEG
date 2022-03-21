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

class CustomDataset(Dataset):
    def __init__(self, clinical, label, eeg):
        self.clinical = clinical
        self.label = label
        self.eeg = eeg

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.clinical[idx], self.label[idx], self.eeg[idx]