import torch
from torch.utils.data import Dataset
import numpy as np

class ClimateDataset(Dataset):
    def __init__(self, xr_dataset, sequence_length=7, target_vars=['rainfall']):
        """
        xr_dataset: Harmonized xarray dataset with time, lat, lon dimensions.
        sequence_length: Number of past days to use as context (T).
        target_vars: List of variables to predict at T+1.
        """
        self.seq_len = sequence_length
        self.target_vars = target_vars
        
        # Dynamically extract list of all variable names (channels) in the dataset
        self.features = list(xr_dataset.data_vars.keys())
        self.num_channels = len(self.features)
        
        # Convert the xarray dataset to a dense numpy array: (Time, Channels, Height, Width)
        data_list = [xr_dataset[var].values for var in self.features]
        self.data = np.stack(data_list, axis=1) 
        
        # Replace NaNs with 0 (Assuming preprocessing handles proper imputation/normalization)
        self.data = np.nan_to_num(self.data, nan=0.0)
        
        # Identify indices for target variables (what we want to predict)
        self.target_indices = [self.features.index(v) for v in self.target_vars]
        
        # Total valid sliding window samples
        self.num_samples = self.data.shape[0] - self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Context Window (X) shape: (Sequence, Channels, Height, Width)
        X = self.data[idx : idx + self.seq_len]
        
        # Target (y) shape: (Target_Channels, Height, Width) at T+1
        y = self.data[idx + self.seq_len, self.target_indices, :, :]
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
