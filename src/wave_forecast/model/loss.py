import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """A loss function combining MSE with physics-based penalties."""
    def __init__(self, mse_weight=1.0, physics_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target, input_features):
        # prediction shape: (batch, seq_len, 3) -> Hs, Tp, MWD
        # target shape: (batch, pred_horizon, 3)
        
        # For simplicity, we predict the last step of the horizon
        pred_last_step = prediction[:, -1, :]
        target_last_step = target[:, -1, :]
        
        # Main MSE Loss
        loss_mse = self.mse_loss(pred_last_step, target_last_step)
        
        # Physics Consistency Penalty (Example: predicted Hs should be consistent with energy)
        # This part can be expanded with more complex checks
        input_energy = input_features[:, -1, 0] # total_energy from the last input step
        predicted_hs = pred_last_step[:, 0]
        
        # Energy should be proportional to Hs^2
        energy_from_hs = (predicted_hs ** 2) / 16.0
        
        # We want the ratio to be close to what it was in the input
        # This is a simplified constraint for demonstration
        loss_physics = self.mse_loss(energy_from_hs, input_energy)
        
        # Total Loss
        total_loss = self.mse_weight * loss_mse + self.physics_weight * loss_physics
        return total_loss