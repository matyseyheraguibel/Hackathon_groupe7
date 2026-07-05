import torch
import torch.nn as nn

class EconomicCriticLoss(nn.Module):
    def __init__(self, trained_critic_model, alpha=1.0, beta=0.1):
        """
        Loss personnalisée combinant précision de pricing (MSE) et rentabilité économique.
        
        Paramètres :
        - trained_critic_model : Le modèle MLP préalablement entraîné sur le critic_dataset.
        - alpha : Poids accordé à la fidélité des prix (MSE).
        - beta : Poids accordé à la rentabilité économique. À augmenter avec précaution !
        """
        super().__init__()
        self.critic = trained_critic_model
        

        for param in self.critic.parameters():
            param.requires_grad = False
            
        # On force le Critique en mode évaluation (désactive le Dropout éventuel)
        self.critic.eval()
        
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        """
        y_pred : Tenseur des prix prédits (Batch, 24)
        y_true : Tenseur des vrais prix (Batch, 24)
        """
        # 1. Calcul de l'erreur absolue classique (Pricing)
        loss_mse = self.mse_loss(y_pred, y_true)
        
        # 2. Préparation de l'entrée pour le Critique
        # Le Critique a été entraîné sur 48 valeurs : [24 vrais prix, 24 prix prédits]
        critic_input = torch.cat([y_true, y_pred], dim=1)
        
        # 3. Évaluation économique par le Critique
        # Renvoie un tenseur de shape (Batch, 1) représentant le taux de capture estimé [0, 1]
        expected_capture_rate = self.critic(critic_input).squeeze()
        
        # 4. Calcul du score économique moyen sur le batch
        # On veut MAXIMISER le taux de capture, donc on le SOUSTRAIT à la perte.
        mean_economic_score = expected_capture_rate.mean()
        
        # 5. Loss totale composée
        total_loss = (self.alpha * loss_mse) - (self.beta * mean_economic_score)
        
        return total_loss