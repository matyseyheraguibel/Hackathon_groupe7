import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """ Initialise un dataset compatible avec les dataLoaders PyTorch afin de faciliter la réalisation des .fit sur des modèles basés en PyTorch"""
        X_np = getattr(X_pandas_or_numpy, "values", X_pandas_or_numpy)
        y_np = getattr(y_pandas_or_numpy, "values", y_pandas_or_numpy)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
