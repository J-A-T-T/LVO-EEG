# Import libraries
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# create custom dataset class
class CustomClinicalDataset(Dataset):
    def __init__(self, clinical, labels):
        self.labels = labels
        self.clinical = clinical

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.clinical[idx,:]
        sample = {"Clinical": data, "Class": label}
        return sample