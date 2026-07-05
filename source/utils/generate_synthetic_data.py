import pandas as pd
import numpy as np
import os
from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.performance_metric import optimal_strategy, strategy_profit

def generate_critic_dataset(
    model, 
    df_X, df_Y,
    coef_path: str, 
    output_path: str,
    noise_levels: list = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
):
    """
    Génère un dataset d'entraînement pour le MLP Critique en ajoutant du bruit blanc
    sur les variables explicatives (X), en générant les prédictions via le modèle, 
    et en calculant le taux de capture journalier.
    """
    
    
    cols_to_exclude = ["day_of_week_target", "cos_week_target", "sin_week_target"]
    cols_to_noise = [c for c in df_X.columns if c not in cols_to_exclude]

    # On a besoin des vrais prix pour calculer l'oracle
    df_y_real = get_pred_df_inverted(df_X, df_Y.values, coef_path)
    y_real_values = df_y_real.values
    
    dataset_rows = []
    
    print(f"Génération du dataset sur {len(df_X)} jours avec {len(noise_levels)} niveaux de bruit...")
    
    # 3. Boucle sur les niveaux d'intensité du bruit
    for sigma in noise_levels:
        print(f"Traitement avec écart-type du bruit (sigma) = {sigma}")
        X_noisy = df_X.copy()
        
        if sigma > 0:
            # Ajout du bruit blanc gaussien sur les colonnes continues
            bruit = np.random.normal(0, sigma, size=(len(df_X), len(cols_to_noise)))
            X_noisy[cols_to_noise] = X_noisy[cols_to_noise] + bruit
            
        # 4. Prédiction avec le modèle de Pricing entraîné sur le X bruité
        # (Attention : model doit implémenter une méthode .predict())
        y_pred_scaled = model.predict(X_noisy)
        
        # 5. Inversion des prédictions bruitées en EUR
        df_pred_real = get_pred_df_inverted(df_X, y_pred_scaled, coef_path)
        y_pred_real_values = df_pred_real.values
        
        # 6. Évaluation jour par jour pour créer les lignes du dataset
        for i in range(len(df_X)):
            true_prices = y_real_values[i]
            pred_prices = y_pred_real_values[i]
            
            # Calcul du profit max possible (Oracle) et du profit simulé
            oracle_profit = strategy_profit(optimal_strategy(true_prices), true_prices)
            realized_profit = strategy_profit(optimal_strategy(pred_prices), true_prices)
            
            # Gestion des jours sans volatilité (pas d'arbitrage possible)
            if oracle_profit <= 0:
                capture_rate = 1.0  # Si l'oracle ne gagne rien, ne rien perdre est un succès
            else:
                capture_rate = realized_profit / oracle_profit
                
            # Assemblage de la ligne : 24 vrais prix + 24 prix prédits + target
            row = list(true_prices) + list(pred_prices) + [capture_rate]
            dataset_rows.append(row)
            
    # 7. Construction du DataFrame final
    true_cols = [f"true_h{h:02d}" for h in range(24)]
    pred_cols = [f"pred_h{h:02d}" for h in range(24)]
    col_names = true_cols + pred_cols + ["capture_rate"]
    
    df_critic = pd.DataFrame(dataset_rows, columns=col_names)
    
    # Sécurité : Écrêter le taux de capture entre 0 et 1 au cas où le modèle fasse n'importe quoi 
    # et tombe sur des edge cases de rentabilité négative extrême.
    df_critic["capture_rate"] = df_critic["capture_rate"].clip(lower=0.0, upper=1.0)
    
    # 8. Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_critic.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé : {output_path} ({df_critic.shape[0]} exemples)")
    
    return df_critic
