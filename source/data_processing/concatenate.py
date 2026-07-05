import pandas as pd
import numpy as np


def concat(df, k):
    # 1. On isole uniquement les colonnes de paramètres numériques à inclure (taille p)
    colonnes_parametres = [col for col in df.columns if col != 'date_paris']
    p = len(colonnes_parametres)
    N = len(df)
    
    # Passage sur NumPy pour maximiser les performances
    matrice_data = df[colonnes_parametres].to_numpy()  # Forme (N, p)
    
    # 2. Extraire les fenêtres glissantes et les empiler en lignes
    lignes_nouvelles = []
    
    for t in range(k, N):
        # On prend le bloc des k heures précédentes : de t-k à t (exclut t)
        bloc = matrice_data[t-k:t]
        
        # .flatten() transforme le bloc (k, p) en un vecteur plat de taille (k * p)
        vecteur_ligne = bloc.flatten()
        lignes_nouvelles.append(vecteur_ligne)
    
    # 3. Créer le nouveau DataFrame (N - k lignes, k * p colonnes)
    # np.vstack empile nos vecteurs pour former les lignes du tableau
    matrice_finale = np.vstack(lignes_nouvelles)
    df_sequence = pd.DataFrame(matrice_finale)
    
    # 4. Optionnel : Nommer proprement les colonnes pour s'y retrouver
    # Exemple : param1_H-30, param2_H-30, ..., paramP_H-1
    noms_colonnes = []
    for h in range(k, 0, -1):  # De l'heure la plus ancienne (k) à la plus récente (1)
        for col in colonnes_parametres:
            noms_colonnes.append(f"{col}_H-{h}")
    
    df_sequence.columns = noms_colonnes #(##### PEUT ETRE)
    return df_sequence
"""
print("Ancienne forme (N, p) :", df[colonnes_parametres].shape)
print("Nouvelle forme requise (N-k, k*p) :", df_sequence.shape)

print(df_sequence.head(1))
"""