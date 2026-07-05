import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .priceModel import PriceModel
from typing import List


class MLPPredictor(nn.Module):
    def __init__(self, num_cont_features, hidden_sizes: List[int], embedding_dim=4):
        super().__init__()
        
        self.day_emb = nn.Embedding(num_embeddings=8, embedding_dim=embedding_dim)
        
        # Construction dynamique du réseau
        layers = []
        # La taille d'entrée est = Nb de features continues + Taille de l'embedding du jour
        input_dim = num_cont_features + embedding_dim
        
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
            
        
        layers.append(nn.Linear(input_dim, 24))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_day, x_cont):
        # On vectorise le jour
        day_encoded = self.day_emb(x_day)

        x = torch.cat([day_encoded, x_cont], dim=1)

        return self.mlp(x)

class MLPPriceModel(PriceModel):
    def __init__(self,
                 hidden_sizes: List[int],
                 weight_decay = 0,
                 embedding_dim=4, 
                 lr=1e-3, 
                 batch_size=32, 
                 epochs=200, 
                 patience=15):

        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None

        # Hardware acceleration cjeck
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.day_col = "day_of_week_target"

    def _prepare_tensors(self, X, y=None):
        """Sépare la colonne jour du reste, et convertit en Tenseurs PyTorch."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Extraction catégorielle (obligatoire en type Long pour nn.Embedding)
        x_day = torch.tensor(X_df[self.day_col].values, dtype=torch.long)
        
        # Extraction continue
        X_cont_df = X_df.drop(columns=[self.day_col])
        x_cont = torch.tensor(X_cont_df.values, dtype=torch.float32)
        
        if y is not None:
            y_df = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y
            y_tensor = torch.tensor(y_df.values, dtype=torch.float32)
            return x_day, x_cont, y_tensor
            
        return x_day, x_cont

    def fit(self, X, y, eval_set=None, metric_funs=[]):
        # Prepare data
        x_day_train, x_cont_train, y_train = self._prepare_tensors(X, y)
        train_dataset = TensorDataset(x_day_train, x_cont_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if eval_set is not None:
            X_val, y_val = eval_set
            x_day_val, x_cont_val, y_val_tensor = self._prepare_tensors(X_val, y_val)

        num_cont_features = x_cont_train.shape[1]
        self.model = MLPPredictor(num_cont_features, self.hidden_sizes, self.embedding_dim).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_day, batch_cont, batch_y in train_loader:
                batch_day, batch_cont, batch_y = batch_day.to(self.device), batch_cont.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_day, batch_cont)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_day.size(0)
                
            train_loss /= len(train_loader.dataset)

            if eval_set is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_preds = self.model(x_day_val.to(self.device), x_cont_val.to(self.device))
                        val_loss = criterion(val_preds, y_val_tensor.to(self.device)).item()
                        
                    if epoch % 10 == 0 or epoch == 0:
                        print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                        
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        self.best_weights = self.model.state_dict() # Sauvegarde les meilleurs poids
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            print(f"Early stopping déclenché à l'epoch {epoch}. Restauration des meilleurs poids.")
                            self.model.load_state_dict(self.best_weights)
                            break

    def predict(self, X):
            self.model.eval()
            x_day, x_cont = self._prepare_tensors(X)
            with torch.no_grad():
                preds = self.model(x_day.to(self.device), x_cont.to(self.device))
                
            return preds.cpu().numpy()