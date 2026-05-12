"""
ClimateResilienceNet — multi-task deep learning model for climate adaptation prediction.

Architecture overview
---------------------
                         Input (N, F)
                              │
                   ┌──────────▼──────────┐
                   │  Feature Tokenizer   │  — projects each feature into d_model dim space
                   │  (Linear + BN + GELU)│    giving a "token" per feature group
                   └──────────┬──────────┘
                              │  (N, n_tokens, d_model)
                   ┌──────────▼──────────┐
                   │ Physics-Informed     │  — soft inductive bias: encodes IPCC zone
                   │ Prior Injection      │    multipliers as learned residual signal
                   └──────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │    4× Transformer Encoder     │
              │  ┌────────────────────────┐   │
              │  │ Pre-LN Self-Attention  │   │
              │  │ (n_heads=8, d_k=16)    │   │
              │  ├────────────────────────┤   │
              │  │ Feed-Forward           │   │
              │  │ (d_ff=512, GELU, Drop) │   │
              │  └────────────────────────┘   │
              └───────────────┬───────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Global Avg Pool +  │
                   │  CLS token          │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Shared Trunk      │
                   │ Linear→BN→SELU→Drop │
                   └──────────┬──────────┘
                    ┌─────────┴──────────┐
          ┌─────────┤    4 Task Heads    ├─────────┐
          │         └────────────────────┘         │
    ResilienceHead  TempReductHead  FloodHead  EnergyHead
     (Sigmoid×100)   (Softplus)     (Sig×90)  (μ + log σ²)

Total params ≈ 520,000
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """
    Projects each input feature (scalar) into a d_model-dimensional embedding.
    Continuous features are projected via a linear layer; the bias acts as a
    learned per-feature offset — similar to FT-Transformer (Gorishniy et al. 2021).
    """

    def __init__(self, n_features: int, d_model: int = 128):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model
        # Per-feature weight: (n_features, d_model) — each feature gets its own projection
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        self.norm   = nn.LayerNorm(d_model)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  →  tokens: (B, F, d_model)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.norm(tokens)


class PhysicsInformedPrior(nn.Module):
    """
    Injects climate-science domain knowledge as a learned residual signal.

    Encodes the IPCC zone multipliers (tropical/subtropical/temperate/cold)
    and intervention interaction priors as trainable embeddings that are
    added to the token sequence before the transformer blocks. This gives
    the model a warm start aligned with physical intuition.

    Rationale for design choice
    ---------------------------
    Purely data-driven models can violate physical constraints (e.g., predicting
    that removing a flood wall increases resilience). The prior embedding
    encodes the direction of effect for each zone×intervention pair, providing
    a soft constraint that improves sample efficiency and physical plausibility.
    """

    # Pre-computed IPCC-based prior magnitudes (relative to temperate baseline)
    ZONE_PRIORS = {
        "tropical":    {"temp_mult": 1.30, "flood_mult": 1.50, "heat_mult": 1.80},
        "subtropical": {"temp_mult": 1.40, "flood_mult": 1.30, "heat_mult": 2.00},
        "temperate":   {"temp_mult": 1.00, "flood_mult": 1.00, "heat_mult": 1.20},
        "cold":        {"temp_mult": 1.60, "flood_mult": 0.70, "heat_mult": 0.60},
    }

    def __init__(self, d_model: int = 128, n_zones: int = 4, n_interventions: int = 4):
        super().__init__()
        self.d_model = d_model
        # Zone embedding — learns to refine the IPCC prior
        self.zone_embed = nn.Embedding(n_zones, d_model)
        # Intervention interaction embedding (2^4 = 16 combos, encoded as int)
        self.intervention_embed = nn.Embedding(16, d_model)
        # Year-horizon prior (discretised into 4 buckets: 2030,2050,2070,2100)
        self.horizon_embed = nn.Embedding(4, d_model)

        self.norm  = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.tensor(0.1))  # Starts small; learned to grow

        self._init_with_ipcc_priors()

    def _init_with_ipcc_priors(self):
        """Warm-initialise zone embeddings from IPCC AR6 multipliers."""
        priors = torch.zeros(4, self.d_model)
        for i, zone in enumerate(["tropical", "subtropical", "temperate", "cold"]):
            p = self.ZONE_PRIORS[zone]
            # Encode three physical signals into the first 3 dims then replicate
            signal = torch.tensor([p["temp_mult"], p["flood_mult"], p["heat_mult"]])
            priors[i, :3] = signal - 1.0  # Deviation from temperate baseline
            # Replicate pattern across d_model with decreasing magnitude
            for j in range(3, self.d_model, 3):
                end = min(j + 3, self.d_model)
                n   = end - j
                priors[i, j:end] = signal[:n] * (0.5 ** (j // 3)) - 1.0
        with torch.no_grad():
            self.zone_embed.weight.copy_(priors)

    def forward(self,
                tokens: torch.Tensor,
                zone_idx: torch.Tensor,
                intervention_combo: torch.Tensor,
                horizon_idx: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, F, d_model)
        Returns tokens + physics prior broadcast over the feature dimension.
        """
        prior = (
            self.zone_embed(zone_idx)               # (B, d_model)
            + self.intervention_embed(intervention_combo)
            + self.horizon_embed(horizon_idx)
        ).unsqueeze(1)  # (B, 1, d_model)

        return tokens + self.scale * self.norm(prior)


class MultiHeadSelfAttention(nn.Module):
    """
    Pre-LayerNorm multi-head self-attention.

    Pre-LN (Xiong et al. 2020) is chosen over post-LN because it provides
    stable gradients without learning-rate warmup, which matters here since
    we combine physics priors with learned attention weights in the same pass.

    n_heads=8, d_model=128  →  d_k = d_v = 16 per head.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.scale   = math.sqrt(self.d_k)

        self.norm    = nn.LayerNorm(d_model)
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = torch.matmul(attn, V)  # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.out(out)


class FeedForward(nn.Module):
    """
    Pre-LN position-wise FFN.
    d_ff = 4 × d_model (standard transformer ratio).
    GELU activation (Hendrycks & Gimpel 2016) — smoother than ReLU for
    tabular data because feature distributions are approximately Gaussian.
    """

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.dropout(self.act(self.linear1(x)))
        return residual + self.linear2(x)


class TransformerEncoderBlock(nn.Module):
    """One pre-LN transformer encoder block: MHSA + FFN."""

    def __init__(self, d_model: int = 128, n_heads: int = 8,
                 d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn  = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class TaskHead(nn.Module):
    """
    Per-task regression head with optional uncertainty estimation.

    Outputs are in NORMALISED (z-score) space — the model trains entirely in
    the preprocessor's scaled space and inverse_transform_y converts back to
    physical units at inference time. Output activations that constrain to
    physical ranges (sigmoid×100, softplus) are intentionally omitted here
    because they conflict with z-scored targets during training.

    For energy_savings we predict both μ and log σ² (aleatoric uncertainty).
    """

    def __init__(self, d_in: int, predict_variance: bool = False):
        super().__init__()
        self.predict_variance = predict_variance

        self.layers = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2 if predict_variance else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # raw z-score output


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ClimateResilienceNet(nn.Module):
    """
    Multi-task transformer for climate resilience prediction.

    Predicts four adaptation outcomes simultaneously:
      1. resilience_score      (0–100)   Sigmoid × 100
      2. temp_reduction_f      (0–15 °F) Softplus
      3. flood_risk_reduction  (0–90 %)  Sigmoid × 90
      4. energy_savings_usd    (0–∞ $/yr) Softplus + uncertainty σ

    Ablation-configurable via constructor flags — used by ablation.py
    to disable individual components and measure their contribution.

    Parameters
    ----------
    n_features         : number of input features
    d_model            : token embedding dimension
    n_heads            : attention heads
    n_layers           : transformer encoder depth
    d_ff               : feed-forward inner dimension
    dropout            : attention & FFN dropout rate
    use_physics_prior  : whether to inject PhysicsInformedPrior
    use_attention      : if False, replaces transformer with MLP (ablation)
    use_uncertainty    : predict aleatoric uncertainty on energy head
    """

    def __init__(
        self,
        n_features:         int   = 36,
        d_model:            int   = 128,
        n_heads:            int   = 8,
        n_layers:           int   = 4,
        d_ff:               int   = 512,
        dropout:            float = 0.10,
        use_physics_prior:  bool  = True,
        use_attention:      bool  = True,
        use_uncertainty:    bool  = True,
    ):
        super().__init__()
        self.n_features        = n_features
        self.d_model           = d_model
        self.use_physics_prior = use_physics_prior
        self.use_attention     = use_attention
        self.use_uncertainty   = use_uncertainty

        # --- Tokenizer ---
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        # CLS token (global representation anchor, like BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Physics prior ---
        if use_physics_prior:
            self.physics_prior = PhysicsInformedPrior(d_model)

        # --- Encoder backbone ---
        if use_attention:
            self.encoder = nn.ModuleList([
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
        else:
            # Ablation: plain MLP backbone (no attention, no feature interaction)
            layers = []
            in_dim = n_features
            for _ in range(n_layers):
                layers += [nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout)]
                in_dim = d_model
            self.encoder = nn.Sequential(*layers)

        self.final_norm = nn.LayerNorm(d_model)

        # --- Shared trunk ---
        trunk_in = d_model if use_attention else d_model
        self.shared_trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # --- Task-specific heads ---
        self.head_resilience  = TaskHead(128)
        self.head_temp        = TaskHead(128)
        self.head_flood       = TaskHead(128)
        self.head_energy      = TaskHead(128, predict_variance=use_uncertainty)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _extract_prior_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract zone, intervention combo, and horizon indices from feature vector.
        Feature layout (from preprocessor):
          zone cols are at positions 13-16 (one-hot)
          intervention cols at 20-23
          year_norm at position 17
        These indices match ALL_FEATURE_GROUPS ordering in preprocessor.py.
        """
        # Zone index: argmax of the 4 one-hot zone columns
        zone_start = 13
        zone_oh    = x[:, zone_start: zone_start + 4]
        zone_idx   = zone_oh.argmax(dim=1).long()

        # Intervention combo: 4-bit binary → integer 0–15
        iv_start  = 20
        iv_bits   = x[:, iv_start: iv_start + 4].round().long()
        combo_idx = (iv_bits[:, 0] * 8 + iv_bits[:, 1] * 4 +
                     iv_bits[:, 2] * 2 + iv_bits[:, 3]).clamp(0, 15)

        # Horizon: year_norm ∈ [0,1] → 4 buckets (roughly 2030/2050/2070/2100)
        year_norm  = x[:, 17]
        horizon    = (year_norm * 3.99).long().clamp(0, 3)

        return zone_idx, combo_idx, horizon

    def _encode_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x   = torch.cat([cls, tokens], dim=1)     # (B, F+1, d_model)
        for block in self.encoder:
            x = block(x)
        x = self.final_norm(x)
        # Use CLS + mean pool for robustness (avoids collapse to single token)
        cls_out  = x[:, 0]
        mean_out = x[:, 1:].mean(dim=1)
        return 0.5 * cls_out + 0.5 * mean_out     # (B, d_model)

    def _encode_mlp(self, x: torch.Tensor) -> torch.Tensor:
        # Ablation path: flat feature vector through MLP
        return self.encoder(x)                     # (B, d_model)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, F) normalised feature tensor

        Returns
        -------
        dict with keys:
          'resilience'  — (B, 1)
          'temp'        — (B, 1)
          'flood'       — (B, 1)
          'energy'      — (B, 1) or (B, 2) if use_uncertainty
        """
        if self.use_attention:
            tokens = self.tokenizer(x)             # (B, F, d_model)

            if self.use_physics_prior:
                z_idx, c_idx, h_idx = self._extract_prior_indices(x)
                tokens = self.physics_prior(tokens, z_idx, c_idx, h_idx)

            h = self._encode_attention(tokens)     # (B, d_model)
        else:
            h = self._encode_mlp(x)                # (B, d_model)

        trunk = self.shared_trunk(h)               # (B, 128)

        return {
            "resilience": self.head_resilience(trunk),
            "temp":       self.head_temp(trunk),
            "flood":      self.head_flood(trunk),
            "energy":     self.head_energy(trunk),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ClimateResilienceNet",
            "=" * 60,
            f"  n_features        : {self.n_features}",
            f"  d_model           : {self.d_model}",
            f"  use_physics_prior : {self.use_physics_prior}",
            f"  use_attention     : {self.use_attention}",
            f"  use_uncertainty   : {self.use_uncertainty}",
            f"  Trainable params  : {self.count_parameters():,}",
            "=" * 60,
        ]
        return "\n".join(lines)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference-mode forward (no grad, eval mode)."""
        self.eval()
        return self.forward(x)

    def get_attention_weights(self, x: torch.Tensor) -> Optional[List[torch.Tensor]]:
        """Returns per-layer attention maps for explainability visualisations."""
        if not self.use_attention:
            return None
        tokens = self.tokenizer(x)
        if self.use_physics_prior:
            z, c, h = self._extract_prior_indices(x)
            tokens = self.physics_prior(tokens, z, c, h)
        B = tokens.shape[0]
        x_seq = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        attn_maps = []
        for block in self.encoder:
            # Re-extract attention weights (requires a slight forward re-trace)
            norm_x = block.attn.norm(x_seq)
            Q = block.attn.q_proj(norm_x)
            K = block.attn.k_proj(norm_x)
            n_heads, d_k = block.attn.n_heads, block.attn.d_k
            Bsz, T, D = Q.shape
            Q_ = Q.view(Bsz, T, n_heads, d_k).transpose(1, 2)
            K_ = K.view(Bsz, T, n_heads, d_k).transpose(1, 2)
            attn = F.softmax(torch.matmul(Q_, K_.transpose(-2, -1)) / math.sqrt(d_k), dim=-1)
            attn_maps.append(attn.detach().cpu())
            x_seq = block(x_seq)
        return attn_maps


# ---------------------------------------------------------------------------
# Ablation variant factory
# ---------------------------------------------------------------------------

def build_model_variant(variant: str, n_features: int) -> ClimateResilienceNet:
    """
    Factory for ablation study variants.

    Variants
    --------
    'full'              : Full ClimateResilienceNet (all components)
    'no_physics_prior'  : Remove IPCC physics prior injection
    'no_attention'      : Replace transformer with plain MLP
    'no_uncertainty'    : Disable aleatoric uncertainty on energy head
    'shallow'           : 2-layer transformer instead of 4
    'narrow'            : d_model=64 instead of 128
    """
    defaults = dict(n_features=n_features, d_model=128, n_heads=8,
                    n_layers=4, d_ff=512, dropout=0.1,
                    use_physics_prior=True, use_attention=True, use_uncertainty=True)

    overrides = {
        "full":             {},
        "no_physics_prior": {"use_physics_prior": False},
        "no_attention":     {"use_attention": False, "use_physics_prior": False},
        "no_uncertainty":   {"use_uncertainty": False},
        "shallow":          {"n_layers": 2},
        "narrow":           {"d_model": 64, "d_ff": 256, "n_heads": 4},
    }
    if variant not in overrides:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(overrides)}")

    cfg = {**defaults, **overrides[variant]}
    return ClimateResilienceNet(**cfg)
