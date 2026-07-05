import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .priceModel import PriceModel
from typing import List
import sys
import os
from source.utils.get_pred_df_inverted import get_pred_df_inverted

# Imports des utils depuis source/utils/
_utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)

from performance_metric import eval_performance

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)


class MLPPredictor(nn.Module):
    def __init__(self, num_cont_features, hidden_sizes: List[int], embedding_dim=4):
        super().__init__()

        self.day_emb = nn.Embedding(num_embeddings=8, embedding_dim=embedding_dim)

        layers = []
        input_dim = num_cont_features + embedding_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, 24))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_day, x_cont):
        day_encoded = self.day_emb(x_day)
        x = torch.cat([day_encoded, x_cont], dim=1)
        return self.mlp(x)


class MLPPriceModelwES(PriceModel):
    """
    MLP avec early stopping base sur la performance economique (eval_performance)
    plutot que sur la val loss MSE.

    Le modele s'entraine avec MSELoss (differentiable) mais selectionne les
    meilleurs poids selon le score d'arbitrage batterie calcule sur le val set.

    La de-standardisation est faite en interne une seule fois au debut du fit,
    garantissant que le score affiche pendant l'entrainement est identique
    au score calcule apres par runmleonardp.py.

    Parametre supplementaire par rapport au MLP classique :
        scaler_params_path : chemin vers data/processed/scaler_params.csv
    """

    def __init__(self,
                 hidden_sizes: List[int],
                 scaler_params_path: str,
                 weight_decay=0,
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
        self.scaler_params_path = scaler_params_path
        self.model = None
        self.best_weights = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.day_col = "day_of_week_target"

    def _prepare_tensors(self, X, y=None):
        """Separe la colonne jour du reste et convertit en tenseurs PyTorch."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        x_day = torch.tensor(X_df[self.day_col].values, dtype=torch.long)

        X_cont_df = X_df.drop(columns=[self.day_col])
        x_cont = torch.tensor(X_cont_df.values, dtype=torch.float32)

        if y is not None:
            y_df = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y
            y_tensor = torch.tensor(y_df.values, dtype=torch.float32)
            return x_day, x_cont, y_tensor

        return x_day, x_cont

    def _load_scaler_for_index(self, index):
        """
        Charge les coefficients rolling_mean / rolling_std alignes sur un index de dates.
        Reproduit exactement la logique de get_pred_df_inverted :
          - decalage d'un jour (on veut les stats du Jour N pour predire J+1)
          - ffill pour les jours manquants
        Retourne mu (n, 1) et sigma (n, 1) en numpy.
        """
        df_coef = pd.read_csv(self.scaler_params_path)
        first_col = df_coef.columns[0]
        df_coef[first_col] = pd.to_datetime(df_coef[first_col].astype(str).str[:10])
        df_coef = df_coef.groupby(first_col).mean()

        target_dates = pd.to_datetime(index).normalize()
        full_index = df_coef.index.union(target_dates)
        df_coef_extended = df_coef.reindex(full_index).ffill()

        dates_pour_coef = target_dates - pd.Timedelta(days=1)
        mu    = df_coef_extended.loc[dates_pour_coef, ['rolling_mean']].values  # (n, 1)
        sigma = df_coef_extended.loc[dates_pour_coef, ['rolling_std']].values   # (n, 1)
        return mu, sigma

    def _destandardize(self, preds_np, mu, sigma):
        """Applique pred * sigma + mu et retourne un DataFrame (n_jours, 24)."""
        prices = (preds_np * sigma) + mu
        return pd.DataFrame(prices, columns=[f"pred_real_h_{i:02d}" for i in range(24)])

    def fit(self, X, y, eval_set=None, metric_funs=[]):
        x_day_train, x_cont_train, y_train = self._prepare_tensors(X, y)
        train_dataset = TensorDataset(x_day_train, x_cont_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        X_val, y_val = None, None
        mu_val, sigma_val = None, None
        y_val_eur = None

        if eval_set is not None:
            X_val, y_val = eval_set
            x_day_val, x_cont_val, y_val_tensor = self._prepare_tensors(X_val, y_val)
            y_val_np = y_val.values if isinstance(y_val, pd.DataFrame) else np.array(y_val)

            # Chargement des coefficients une seule fois, identique a get_pred_df_inverted
            #mu_val, sigma_val = self._load_scaler_for_index(X_val.index)

            # Vraies valeurs en EUR, calculees une seule fois
            #y_val_eur = self._destandardize(y_val_np, mu_val, sigma_val)
            y_val_eur = get_pred_df_inverted(X_val, y_val_np, self.scaler_params_path)

        num_cont_features = x_cont_train.shape[1]
        self.model = MLPPredictor(num_cont_features, self.hidden_sizes, self.embedding_dim).to(self.device)

        criterion = RMSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_score = float('-inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_day, batch_cont, batch_y in train_loader:
                batch_day  = batch_day.to(self.device)
                batch_cont = batch_cont.to(self.device)
                batch_y    = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_day, batch_cont)
                loss  = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_day.size(0)

            train_loss /= len(train_loader.dataset)

            if eval_set is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds_tensor = self.model(
                        x_day_val.to(self.device),
                        x_cont_val.to(self.device)
                    )
                    val_loss = criterion(val_preds_tensor, y_val_tensor.to(self.device)).item()

                # De-standardisation avec les coefficients pre-charges (meme logique que runmleonardp)
                #val_preds_np  = val_preds_tensor.cpu().numpy()
                #val_preds_eur = self._destandardize(val_preds_np, mu_val, sigma_val)
                val_preds_np  = val_preds_tensor.cpu().numpy()
                val_preds_eur = get_pred_df_inverted(X_val, val_preds_np, self.scaler_params_path)
                

                # Score economique — calcul identique a celui de affichage_metrics
                val_score = eval_performance(y_val_eur, val_preds_eur)

                if epoch % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f} | Score eco: {val_score:.4f}")

                if val_score > best_val_score:
                    best_val_score = val_score
                    epochs_no_improve = 0
                    self.best_weights = self.model.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping a l'epoch {epoch}. Meilleur score eco: {best_val_score:.4f}. Restauration des meilleurs poids.")
                        self.model.load_state_dict(self.best_weights)
                        break

        if eval_set is None and self.best_weights is None:
            self.best_weights = self.model.state_dict()

    def predict(self, X):
        self.model.eval()
        x_day, x_cont = self._prepare_tensors(X)
        with torch.no_grad():
            preds = self.model(x_day.to(self.device), x_cont.to(self.device))
        return preds.cpu().numpy()
