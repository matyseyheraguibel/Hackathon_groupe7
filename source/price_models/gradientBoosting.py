from .priceModel import PriceModel
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

class GradientBoosting(PriceModel):
    def __init__(self,
        objective="regression",      # Loss (regression = MSE, regression_l1 = MAE)
        n_estimators=100,            # Nombre d'arbres
        learning_rate=0.05,          
        max_depth=-1,              
        num_leaves=31,               
        min_child_samples=20,
        patience = 10,
        categorical_feature=None,    # Si on a des variables qualitatives passer leurs noms ici
        random_state=42,             
        n_jobs=-1,                   # Nb processeurs
        period=10, 
        **kwargs                     # Pour ajouter
    ):
        
        self.model_params = {
            "objective": objective,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
            **kwargs
        }

        self.categorical_feature = categorical_feature
        self.patience = patience
        self.period = period
        base_model = lgb.LGBMRegressor(**self.model_params)
        self.model = MultiOutputRegressor(base_model)

    def fit(self, X, y, eval_set=None, metric_funs = []):
        """ eval_set is either None or (X_val, y_val),
            metric_funs is a list of functions to apply to the couple (Y_pred, Y_test)
        """
        self.models = []

        n_outputs = y.shape[1] if len(y.shape) > 1 else 1
        
        print(f"Début de l'entraînement de {n_outputs} modèles indépendants avec Early Stopping...")

        for i in range(n_outputs):
            model = lgb.LGBMRegressor(**self.model_params)
            
            # On extrait la i-ème colonne pour la cible d'entraînement
            y_train_i = y.iloc[:, i] if hasattr(y, "iloc") else y[:, i]
            
            fit_kwargs = {"categorical_feature": self.categorical_feature}
            
            # Si un eval_set est fourni, on découpe la i-ème colonne
            if eval_set is not None:
                X_val, y_val = eval_set
                y_val_i = y_val.iloc[:, i] if hasattr(y_val, "iloc") else y_val[:, i]
                
                fit_kwargs["eval_set"] = [(X_val, y_val_i)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=self.patience, verbose=False),
                    # period=10 affichera la loss toutes les 10 itérations 
                    lgb.log_evaluation(period=self.period) 
                ]
            
            print(f"\n--- Entraînement Heure {i:02d}h00 ---")
            model.fit(X, y_train_i, **fit_kwargs)

            for i, funs in enumerate(metric_funs):
                print(f"{i}-th eval metric is : {funs(model.predict(X_val), y_val)}")
            
            # On stocke le modèle entraîné pour cette heure
            self.models.append(model)
    

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.column_stack(preds)