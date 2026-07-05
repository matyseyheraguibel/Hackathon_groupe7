import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .priceModel import PriceModel
from typing import List
import os
import sys


_utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if _utils_path not in sys.path:
    sys.path.insert(0, _utils_path)

from performance_metric import eval_performance

class NeuralRouterPredictor(nn.Module):
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
        layers.append(nn.Sigmoid()) 
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_day, x_cont):
        day_encoded = self.day_emb(x_day)
        x = torch.cat([day_encoded, x_cont], dim=1)
        return self.mlp(x) # Retourne un vecteur (batch, 24) de valeurs entre 0 et 1


class NeuralRouterPriceModel(PriceModel):
    def __init__(self, 
                 mlp_model_instance, 
                 gb_model_instance, 
                 scaler_params_path,
                 router_hidden_sizes=[32],
                 embedding_dim=4, 
                 lr=1e-3, 
                 batch_size=64, 
                 epochs=200, 
                 patience=20,
                 weight_decay=1e-4):
        
        self.mlp_model = mlp_model_instance
        self.gb_model = gb_model_instance
        self.scaler_params_path = scaler_params_path
        
        # Hyperparamètres du Routeur
        self.router_hidden_sizes = router_hidden_sizes
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.day_col = "day_of_week_target"
        self.router = None
        self.best_weights = None

    def _prepare_tensors(self, X, y=None):
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

    def fit(self, X_base, y_base, X_meta, y_meta, eval_set=None, metric_funs=[]):
        """
        X_base, y_base : Entraînement brut des modèles de base (ex: < 2025)
        X_meta, y_meta : Set intermédiaire pour entraîner le Routeur (ex: 2025)
        eval_set       : Test final pour l'Early Stopping du Routeur (ex: 2026)
        """
        
        # -------------------------------------------------------------
        # ÉTAPE 1 : ENTRAÎNEMENT DES SOUS-MODÈLES SUR LA BASE
        # -------------------------------------------------------------
        print("--- Étape 1 : Entraînement du modèle MLP ---")
        # On peut utiliser le set Meta pour l'early stopping du MLP
        self.mlp_model.fit(X_base, y_base, eval_set=(X_meta, y_meta))
        
        print("\n--- Étape 2 : Entraînement du modèle Gradient Boosting ---")
        # On peut utiliser le set Meta pour l'early stopping du GB
        self.gb_model.fit(X_base, y_base, eval_set=(X_meta, y_meta))
        
        # -------------------------------------------------------------
        # ÉTAPE 2 : GÉNÉRATION DES PRÉDICTIONS "HONNÊTES" SUR LE SET META
        # -------------------------------------------------------------
        print("\n--- Étape 3 : Entraînement du Réseau de Neurones Routeur ---")
        # CRITIQUE : Le routeur va s'entraîner sur X_meta, des données que les 
        # modèles n'ont pas apprises par cœur.
        mlp_preds_meta = torch.tensor(self.mlp_model.predict(X_meta), dtype=torch.float32)
        gb_preds_meta = torch.tensor(self.gb_model.predict(X_meta), dtype=torch.float32)
        
        x_day_meta, x_cont_meta, y_meta_tensor = self._prepare_tensors(X_meta, y_meta)
        
        # Le dataset du routeur est construit EXCLUSIVEMENT sur X_meta
        train_dataset = TensorDataset(x_day_meta, x_cont_meta, mlp_preds_meta, gb_preds_meta, y_meta_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # -------------------------------------------------------------
        # ÉTAPE 3 : PRÉPARATION DU TEST SET (2026) POUR L'EARLY STOPPING
        # -------------------------------------------------------------
        if eval_set is not None:
            X_test, y_test = eval_set
            x_day_test, x_cont_test, y_test_tensor = self._prepare_tensors(X_test, y_test)
            
            mlp_preds_test = torch.tensor(self.mlp_model.predict(X_test), dtype=torch.float32).to(self.device)
            gb_preds_test = torch.tensor(self.gb_model.predict(X_test), dtype=torch.float32).to(self.device)
            
            y_test_np = y_test.values if isinstance(y_test, pd.DataFrame) else np.array(y_test)
            mu_t, sigma_t = self._load_scaler_for_index(X_test.index)
            y_test_eur = self._destandardize(y_test_np, mu_t, sigma_t)

        num_cont_features = x_cont_meta.shape[1]
        self.router = NeuralRouterPredictor(num_cont_features, self.router_hidden_sizes, self.embedding_dim).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.router.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_score = float('-inf')
        epochs_no_improve = 0

        # -------------------------------------------------------------
        # ÉTAPE 4 : BOUCLE D'ENTRAÎNEMENT DU ROUTEUR
        # -------------------------------------------------------------
        for epoch in range(self.epochs):
            self.router.train()
            train_loss = 0.0
            
            for b_day, b_cont, b_mlp_p, b_gb_p, b_y in train_loader:
                b_day, b_cont = b_day.to(self.device), b_cont.to(self.device)
                b_mlp_p, b_gb_p, b_y = b_mlp_p.to(self.device), b_gb_p.to(self.device), b_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Le routeur prédit alpha (entre 0 et 1)
                alpha = self.router(b_day, b_cont)
                
                # Mélange des prédictions : alpha * MLP + (1 - alpha) * GB
                mixed_preds = alpha * b_mlp_p + (1.0 - alpha) * b_gb_p
                
                # Le routeur ajuste ses poids pour minimiser l'erreur de ce mélange
                loss = criterion(mixed_preds, b_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * b_day.size(0)
                
            train_loss /= len(train_loader.dataset)

            # Évaluation de l'Early Stopping sur 2026 !
            if eval_set is not None:
                self.router.eval()
                with torch.no_grad():
                    alpha_test = self.router(x_day_test.to(self.device), x_cont_test.to(self.device))
                    mixed_preds_test = alpha_test * mlp_preds_test + (1.0 - alpha_test) * gb_preds_test
                    
                val_preds_np = mixed_preds_test.cpu().numpy()
                val_preds_eur = self._destandardize(val_preds_np, mu_t, sigma_t)
                
                # On juge le Routeur sur l'argent réel qu'il génère sur 2026
                val_score = eval_performance(y_test_eur, val_preds_eur)

                if epoch % 10 == 0 or epoch == 0:
                    print(f"Epoch Routeur {epoch:03d}/{self.epochs} | Train Loss (Meta): {train_loss:.4f} | Capture Test (2026): {val_score*100:.2f} %")

                if val_score > best_val_score:
                    best_val_score = val_score
                    epochs_no_improve = 0
                    self.best_weights = self.router.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping du Routeur à l'epoch {epoch}. Capture Max: {best_val_score*100:.2f} %. Restauration.")
                        self.router.load_state_dict(self.best_weights)
                        break
    def predict(self, X):
        # 1. On prédit avec les modèles de base
        mlp_preds = torch.tensor(self.mlp_model.predict(X), dtype=torch.float32).to(self.device)
        gb_preds = torch.tensor(self.gb_model.predict(X), dtype=torch.float32).to(self.device)
        
        # 2. Le routeur analyse X pour prédire alpha
        self.router.eval()
        x_day, x_cont = self._prepare_tensors(X)
        with torch.no_grad():
            alpha = self.router(x_day.to(self.device), x_cont.to(self.device))
            
        # 3. Application du mélange dynamique
        mixed = alpha * mlp_preds + (1.0 - alpha) * gb_preds
        return mixed.cpu().numpy()