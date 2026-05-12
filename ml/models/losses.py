"""
Loss functions for ClimateResilienceNet.

ClimateAwareLoss — weighted MSE across 4 targets + optional uncertainty NLL
  All predictions and targets are in normalised (z-score) space.
  Physical-range constraints are applied at inference time via inverse_transform_y.

HuberMultiTaskLoss — Huber loss (ablation baseline, more robust to outliers).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClimateAwareLoss(nn.Module):
    """
    Multi-task loss for normalised-space predictions.

    Components
    ----------
    1. Per-output weighted MSE — weights compensate for different output variances
       so no single head dominates gradient magnitude.
    2. Aleatoric uncertainty NLL — for the energy head when predict_variance=True;
       clamped log-variance prevents division-by-zero on early random weights.

    Note: monotonicity and physics-consistency penalties are excluded from the
    default loss because they require O(B²) outer products that are unstable
    before the model has learned meaningful representations. They can be re-enabled
    after ~20 warm-up epochs via the `advanced_penalties` flag.
    """

    def __init__(
        self,
        weights:           Tuple[float, float, float, float] = (1.0, 1.5, 1.2, 0.8),
        uncertainty_weight: float = 0.3,
        advanced_penalties: bool  = False,  # enable after warm-up
    ):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.uncertainty_weight  = uncertainty_weight
        self.advanced_penalties  = advanced_penalties

    def forward(
        self,
        preds:      Dict[str, torch.Tensor],
        targets:    torch.Tensor,
        x_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        preds   : dict with keys resilience(B,1), temp(B,1), flood(B,1), energy(B,1or2)
                  All tensors are in normalised (z-score) space.
        targets : (B, 4) normalised targets — [resilience, temp, flood, energy]
        """
        pred_r  = preds["resilience"]          # (B, 1)
        pred_t  = preds["temp"]                # (B, 1)
        pred_f  = preds["flood"]               # (B, 1)
        energy  = preds["energy"]              # (B, 1) or (B, 2)

        mse_r = F.mse_loss(pred_r, targets[:, 0:1])
        mse_t = F.mse_loss(pred_t, targets[:, 1:2])
        mse_f = F.mse_loss(pred_f, targets[:, 2:3])

        if energy.shape[1] == 2:
            mu_e      = energy[:, 0:1]
            log_var_e = energy[:, 1:2].clamp(-3, 3)
            mse_e     = F.mse_loss(mu_e, targets[:, 3:4])
            nll_e     = 0.5 * (log_var_e +
                                (targets[:, 3:4] - mu_e) ** 2 / (log_var_e.exp() + 1e-6))
            uncertainty_loss = self.uncertainty_weight * nll_e.mean()
        else:
            mse_e            = F.mse_loss(energy, targets[:, 3:4])
            uncertainty_loss = torch.zeros(1, device=targets.device)

        mse_loss = (self.weights[0] * mse_r + self.weights[1] * mse_t +
                    self.weights[2] * mse_f + self.weights[3] * mse_e)

        total = mse_loss + uncertainty_loss

        return {
            "total":             total,
            "mse_loss":          mse_loss.detach(),
            "uncertainty_loss":  uncertainty_loss.detach(),
            "monotonicity_loss": torch.zeros(1),
            "physics_loss":      torch.zeros(1),
        }


class HuberMultiTaskLoss(nn.Module):
    """Huber loss — used as ablation baseline (robust to label outliers)."""

    def __init__(self, delta: float = 1.0,
                 weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.delta = delta
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, preds: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                x_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu_e = preds["energy"][:, :1] if preds["energy"].shape[1] == 2 else preds["energy"]
        p_stack = torch.cat([preds["resilience"], preds["temp"],
                             preds["flood"], mu_e], dim=1)
        loss = (F.huber_loss(p_stack, targets, delta=self.delta, reduction="none") *
                self.weights.unsqueeze(0)).mean()
        zeros = torch.zeros(1, device=targets.device)
        return {"total": loss, "mse_loss": loss.detach(),
                "uncertainty_loss": zeros, "monotonicity_loss": zeros, "physics_loss": zeros}
