import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

class CriticMLP(nn.Module):
    def __init__(self, lr=1e-3, batch_size=64, epochs=150, patience=20):
        """
        Modèle d'évaluation économique (Critique).
        Prend 48 variables (24 vrais prix + 24 prix prédits) et renvoie le Taux de Capture.
        """
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Architecture du Critique
        self.net = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(self.device)

    def forward(self, x):
        """
        Définit le passage des données dans le réseau (Forward pass).
        x : Tenseur de shape (Batch, 48)
        """
        return self.net(x)

    def _prepare_tensors(self, X, y=None):
        """Convertit les DataFrames Pandas en Tenseurs PyTorch."""
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        
        if y is not None:
            y_np = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
            y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1) # Force shape (N, 1)
            return X_tensor, y_tensor
            
        return X_tensor

    def fit(self, X, y, eval_set=None):
        """
        Boucle d'entraînement du Critique.
        X : (N, 48) - 24 vrais prix suivis de 24 prix prédits.
        y : (N, 1)  - Taux de capture économique.
        """
        X_train, y_train = self._prepare_tensors(X, y)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if eval_set is not None:
            X_val, y_val = self._prepare_tensors(eval_set[0], eval_set[1])
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)

        # On utilise la MSE car on veut que la prédiction s'approche du vrai pourcentage
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_weights = None
        
        print(f"Début de l'entraînement du Critique sur {self.device}...")
        
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.net(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Phase de validation
            if eval_set is not None:
                self.eval()
                with torch.no_grad():
                    val_preds = self.net(X_val)
                    val_loss = criterion(val_preds, y_val).item()
                    
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:03d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    
                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = self.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping déclenché à l'epoch {epoch+1}.")
                        self.load_state_dict(best_weights)
                        break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:03d}/{self.epochs} | Train Loss: {train_loss:.4f}")

    def predict(self, X):
        """Renvoie le Taux de Capture estimé pour de nouvelles courbes."""
        self.eval()
        X_tensor = self._prepare_tensors(X).to(self.device)
        with torch.no_grad():
            preds = self.net(X_tensor)
        return preds.cpu().numpy()