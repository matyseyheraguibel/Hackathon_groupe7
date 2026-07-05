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

# Imports des utils depuis source/utils/
_utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)

from performance_metric import eval_performance


# =====================================================================
# Structure de X attendue (telle que produite par prepare_dataset) :
#
#   Colonnes contextuelles (features du jour cible N+1) :
#       day_of_week_target  : entier 0-6
#       sin_week_target     : float
#       cos_week_target     : float
#
#   Colonnes de sequence temporelle (ordre chronologique inverse dans X,
#   on les remet dans l'ordre dans _prepare_tensors) :
#       price_N-1_h00 ... price_N-1_h23  : prix du jour N-1 (48h avant)
#       price_N-0_h00 ... price_N-0_h23  : prix du jour N   (24h avant)
#
#   Colonnes de features du jour cible (pred_prod) :
#       pred_prod_h00 ... pred_prod_h23  : prevision de production du jour N+1
#
# Choix d'architecture LSTM :
#   - Sequence : 48 pas de temps (heure par heure de N-1_h00 a N-0_h23)
#   - Features par pas : 1 (prix standardise) + 1 (pred_prod de l'heure correspondante
#     pour les 24 derniers pas, 0 pour les 24 premiers)
#   - Apres le LSTM : on concatene la sortie avec les features contextuelles
#     (sin_week, cos_week) et l'embedding du jour
#   - Sortie : vecteur de 24 valeurs (prix des 24h du jour N+1)
# =====================================================================


class LSTMPredictor(nn.Module):
    def __init__(self,
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 1,
                 embedding_dim: int = 4,
                 dropout_rate: float = 0.0,
                 fc_hidden_sizes: List[int] = None):
        """
        lstm_hidden_size  : taille de l'etat cache du LSTM
        lstm_num_layers   : nombre de couches LSTM empilees
        embedding_dim     : dimension de l'embedding du jour de la semaine
        dropout_rate      : dropout entre les couches LSTM (si num_layers > 1)
        fc_hidden_sizes   : couches fully-connected apres le LSTM (ex: [64, 32])
        """
        super().__init__()

        self.day_emb = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)

        # LSTM : input = 1 feature par pas (le prix), ou 2 si on injecte pred_prod
        # On injecte pred_prod comme 2eme feature sur les 24 derniers pas
        # Pour simplifier on passe 2 features a tous les pas (0 pour pred_prod quand indisponible)
        lstm_input_size = 2  # (prix, pred_prod_ou_zero)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0.0
        )

        # Apres le LSTM on concatene : sortie LSTM + embedding_jour + sin_week + cos_week
        fc_input_size = lstm_hidden_size + embedding_dim + 2  # +2 pour sin et cos

        fc_layers = []
        if fc_hidden_sizes:
            in_dim = fc_input_size
            for h in fc_hidden_sizes:
                fc_layers.append(nn.Linear(in_dim, h))
                fc_layers.append(nn.ReLU())
                in_dim = h
            fc_layers.append(nn.Linear(in_dim, 24))
        else:
            fc_layers.append(nn.Linear(fc_input_size, 24))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x_seq, x_day, x_context):
        """
        x_seq     : (batch, 48, 2)  sequence temporelle (prix + pred_prod)
        x_day     : (batch,)        jour de la semaine (entier)
        x_context : (batch, 2)      [sin_week, cos_week]
        """
        # LSTM - on prend uniquement le dernier etat cache
        lstm_out, _ = self.lstm(x_seq)          # (batch, 48, hidden_size)
        lstm_last = lstm_out[:, -1, :]           # (batch, hidden_size)

        # Embedding du jour
        day_encoded = self.day_emb(x_day)        # (batch, embedding_dim)

        # Concatenation finale
        combined = torch.cat([lstm_last, day_encoded, x_context], dim=1)

        return self.fc(combined)                 # (batch, 24)


class LSTMPriceModel(PriceModel):
    """
    Modele LSTM pour la prevision des prix SPOT J+1.

    Traite la fenetre historique de prix comme une vraie sequence temporelle
    (contrairement au MLP qui la recoit comme un vecteur plat).

    Early stopping base sur la performance economique (eval_performance),
    identique a mleonardp.py.

    Parametres
    ----------
    scaler_params_path : chemin vers data/processed/scaler_params.csv
    lstm_hidden_size   : taille de l'etat cache LSTM (defaut 64)
    lstm_num_layers    : nb de couches LSTM empilees (defaut 1)
    fc_hidden_sizes    : couches FC apres le LSTM, ex [64] (defaut None = une seule couche)
    embedding_dim      : dimension embedding jour semaine (defaut 4)
    dropout_rate       : dropout LSTM, actif seulement si lstm_num_layers > 1
    lr, weight_decay, batch_size, epochs, patience : hyperparametres classiques
    """

    def __init__(self,
                 scaler_params_path: str,
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 1,
                 fc_hidden_sizes: List[int] = None,
                 embedding_dim: int = 4,
                 dropout_rate: float = 0.0,
                 weight_decay: float = 0.0,
                 lr: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 200,
                 patience: int = 20):

        self.scaler_params_path = scaler_params_path
        self.lstm_hidden_size   = lstm_hidden_size
        self.lstm_num_layers    = lstm_num_layers
        self.fc_hidden_sizes    = fc_hidden_sizes
        self.embedding_dim      = embedding_dim
        self.dropout_rate       = dropout_rate
        self.weight_decay       = weight_decay
        self.lr                 = lr
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.patience           = patience
        self.model              = None
        self.best_weights       = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Noms de colonnes attendus dans X
        self.day_col     = "day_of_week_target"
        self.sin_col     = "sin_week_target"
        self.cos_col     = "cos_week_target"
        self.price_prefix    = "price_"
        self.pred_prod_prefix = "pred_prod_h"

    def _prepare_tensors(self, X, y=None):
        """
        Reconstruit la sequence temporelle a partir des colonnes de X.

        Retourne :
            x_seq     : (n, 48, 2)  sequence (prix, pred_prod_ou_zero) ordre chrono
            x_day     : (n,)        entiers 0-6
            x_context : (n, 2)      [sin_week, cos_week]
            y_tensor  : (n, 24)     optionnel
        """
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # --- Features contextuelles ---
        x_day     = torch.tensor(X_df[self.day_col].values, dtype=torch.long)
        x_context = torch.tensor(
            X_df[[self.sin_col, self.cos_col]].values, dtype=torch.float32
        )

        # --- Colonnes de prix : on les remet dans l'ordre chronologique ---
        # Dans X : price_N-1_h00 ... price_N-1_h23, price_N-0_h00 ... price_N-0_h23
        # Ordre chrono : N-1_h00 -> N-1_h23 -> N-0_h00 -> N-0_h23
        price_cols_day1 = [f"price_N-1_h{h:02d}" for h in range(24)]
        price_cols_day0 = [f"price_N-0_h{h:02d}" for h in range(24)]
        price_cols_ordered = price_cols_day1 + price_cols_day0  # 48 pas chrono

        prices_np = X_df[price_cols_ordered].values  # (n, 48)

        # --- Colonnes pred_prod (24h du jour cible) ---
        # On les aligne sur les 24 derniers pas de la sequence (= jour N-0)
        pred_prod_cols = [f"pred_prod_h{h:02d}" for h in range(24)]
        pred_prod_np = X_df[pred_prod_cols].values  # (n, 24)

        # Construction de la sequence (n, 48, 2)
        n = len(X_df)
        seq = np.zeros((n, 48, 2), dtype=np.float32)
        seq[:, :, 0] = prices_np                          # feature 0 : prix
        seq[:, 24:, 1] = pred_prod_np                     # feature 1 : pred_prod sur les 24 derniers pas

        x_seq = torch.tensor(seq, dtype=torch.float32)

        if y is not None:
            y_df = y if isinstance(y, pd.DataFrame) else pd.DataFrame(y)
            y_tensor = torch.tensor(y_df.values, dtype=torch.float32)
            return x_seq, x_day, x_context, y_tensor

        return x_seq, x_day, x_context

    def _load_scaler_for_index(self, index):
        """Meme logique que mleonardp : charge mu/sigma alignes sur l'index."""
        df_coef = pd.read_csv(self.scaler_params_path)
        first_col = df_coef.columns[0]
        df_coef[first_col] = pd.to_datetime(df_coef[first_col].astype(str).str[:10])
        df_coef = df_coef.groupby(first_col).mean()

        target_dates = pd.to_datetime(index).normalize()
        full_index = df_coef.index.union(target_dates)
        df_coef_extended = df_coef.reindex(full_index).ffill()

        dates_pour_coef = target_dates - pd.Timedelta(days=1)
        mu    = df_coef_extended.loc[dates_pour_coef, ['rolling_mean']].values
        sigma = df_coef_extended.loc[dates_pour_coef, ['rolling_std']].values
        return mu, sigma

    def _destandardize(self, preds_np, mu, sigma):
        prices = (preds_np * sigma) + mu
        return pd.DataFrame(prices, columns=[f"pred_real_h_{i:02d}" for i in range(24)])

    def fit(self, X, y, eval_set=None, metric_funs=[]):
        x_seq_train, x_day_train, x_ctx_train, y_train = self._prepare_tensors(X, y)
        train_dataset = TensorDataset(x_seq_train, x_day_train, x_ctx_train, y_train)
        train_loader  = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        mu_val, sigma_val, y_val_eur = None, None, None
        x_seq_val = x_day_val = x_ctx_val = y_val_tensor = None

        if eval_set is not None:
            X_val, y_val = eval_set
            x_seq_val, x_day_val, x_ctx_val, y_val_tensor = self._prepare_tensors(X_val, y_val)
            y_val_np  = y_val.values if isinstance(y_val, pd.DataFrame) else np.array(y_val)
            mu_val, sigma_val = self._load_scaler_for_index(X_val.index)
            y_val_eur = self._destandardize(y_val_np, mu_val, sigma_val)

        self.model = LSTMPredictor(
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            embedding_dim=self.embedding_dim,
            dropout_rate=self.dropout_rate,
            fc_hidden_sizes=self.fc_hidden_sizes,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_score  = float('-inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for b_seq, b_day, b_ctx, b_y in train_loader:
                b_seq = b_seq.to(self.device)
                b_day = b_day.to(self.device)
                b_ctx = b_ctx.to(self.device)
                b_y   = b_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(b_seq, b_day, b_ctx)
                loss  = criterion(preds, b_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * b_seq.size(0)

            train_loss /= len(train_loader.dataset)

            if eval_set is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds_tensor = self.model(
                        x_seq_val.to(self.device),
                        x_day_val.to(self.device),
                        x_ctx_val.to(self.device)
                    )
                    val_loss = criterion(val_preds_tensor, y_val_tensor.to(self.device)).item()

                val_preds_np  = val_preds_tensor.cpu().numpy()
                val_preds_eur = self._destandardize(val_preds_np, mu_val, sigma_val)
                val_score     = eval_performance(y_val_eur, val_preds_eur)

                if epoch % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f} | Score eco: {val_score:.4f}")

                if val_score > best_val_score:
                    best_val_score    = val_score
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
        x_seq, x_day, x_ctx = self._prepare_tensors(X)
        with torch.no_grad():
            preds = self.model(
                x_seq.to(self.device),
                x_day.to(self.device),
                x_ctx.to(self.device)
            )
        return preds.cpu().numpy()
