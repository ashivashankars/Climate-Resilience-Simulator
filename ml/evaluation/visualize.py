"""
Visualization utilities for model metrics, ablation studies, and predictions.

All plots use a consistent dark-mode climate theme matching the web UI.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for server-side rendering

# ── Design constants ──────────────────────────────────────────────────────────
DARK_BG   = "#0f172a"
CARD_BG   = "#1e293b"
ACCENT    = "#06b6d4"   # Cyan
GREEN     = "#22c55e"
AMBER     = "#f59e0b"
RED       = "#ef4444"
TEXT      = "#f1f5f9"
MUTED     = "#94a3b8"
PALETTE   = [ACCENT, GREEN, AMBER, RED, "#8b5cf6", "#ec4899"]

TARGET_DISPLAY = {
    "resilience": "Resilience Score",
    "temp":       "Temp Reduction (°F)",
    "flood":      "Flood Risk Reduction (%)",
    "energy":     "Energy Savings ($/yr)",
}

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD_BG)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color=DARK_BG, linewidth=0.6, alpha=0.8)


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history_path: str, output_dir: str) -> str:
    with open(history_path) as f:
        history = json.load(f)

    df = pd.DataFrame(history)
    epochs = df["epoch"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=DARK_BG)
    fig.suptitle("Training History — ClimateResilienceNet", color=TEXT,
                 fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color=ACCENT,  lw=2, label="Train")
    ax.plot(epochs, df["val_loss"],   color=GREEN,   lw=2, label="Val")
    _style_ax(ax, "Loss", "Epoch", "Loss")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8)

    # Per-output R²
    ax = axes[1]
    target_names = ["resilience", "temp", "flood", "energy"]
    for name, color in zip(target_names, PALETTE):
        col = f"val_r2_{name}"
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, lw=1.8, label=TARGET_DISPLAY[name])
    _style_ax(ax, "Validation R² per Output", "Epoch", "R²")
    ax.axhline(0, color=MUTED, linestyle="--", lw=0.8, alpha=0.5)
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8, fontsize=8)

    # Mean R²
    ax = axes[2]
    if "val_r2_mean" in df.columns:
        ax.plot(epochs, df["val_r2_mean"], color=ACCENT, lw=2)
    _style_ax(ax, "Mean Validation R²", "Epoch", "Mean R²")
    ax.fill_between(epochs, df.get("val_r2_mean", [0]*len(epochs)),
                    alpha=0.15, color=ACCENT)

    plt.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    plt.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return str(out_path)


# ---------------------------------------------------------------------------
# 2. Predicted vs actual scatter
# ---------------------------------------------------------------------------

def plot_predictions(npz_path: str, output_dir: str) -> str:
    data    = np.load(npz_path)
    preds   = data["preds_orig"]
    targets = data["tgts_orig"]
    names   = ["resilience", "temp", "flood", "energy"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=DARK_BG)
    fig.suptitle("Predicted vs Actual — Test Set", color=TEXT,
                 fontsize=13, fontweight="bold")

    for i, (name, ax) in enumerate(zip(names, axes)):
        p, t = preds[:, i], targets[:, i]
        ax.scatter(t, p, alpha=0.25, s=5, color=PALETTE[i])
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.5, linestyle="--")
        r2 = 1 - np.var(p - t) / np.var(t)
        _style_ax(ax, f"{TARGET_DISPLAY[name]}\nR²={r2:.3f}",
                  "Actual", "Predicted")

    plt.tight_layout()
    out_path = Path(output_dir) / "predictions_scatter.png"
    plt.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return str(out_path)


# ---------------------------------------------------------------------------
# 3. Ablation study bar chart
# ---------------------------------------------------------------------------

def plot_ablation(ablation_csv: str, output_dir: str) -> str:
    df = pd.read_csv(ablation_csv, index_col="variant")

    names = ["resilience", "temp", "flood", "energy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("Ablation Study — Component Contribution", color=TEXT,
                 fontsize=13, fontweight="bold")

    # Mean R² bar chart
    ax = axes[0]
    variants = df.index.tolist()
    r2_means = df["r2_mean"].values
    bars = ax.barh(variants[::-1], r2_means[::-1], color=ACCENT, alpha=0.85)
    for bar, val in zip(bars, r2_means[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color=TEXT, fontsize=8)
    _style_ax(ax, "Mean R² by Model Variant", "R²", "Variant")
    ax.set_xlim(0, 1.0)

    # Per-target grouped bar chart for full vs best ablation
    ax = axes[1]
    x  = np.arange(len(names))
    bw = 0.15
    for j, (variant, color) in enumerate(zip(variants[:4], PALETTE)):
        vals = [df.loc[variant, f"r2_{n}"] if f"r2_{n}" in df.columns else 0
                for n in names]
        ax.bar(x + j * bw, vals, width=bw, label=variant, color=color, alpha=0.85)

    ax.set_xticks(x + bw * 1.5)
    ax.set_xticklabels([TARGET_DISPLAY[n] for n in names], fontsize=8, rotation=10)
    ax.set_ylim(0, 1.05)
    _style_ax(ax, "Per-Output R² Comparison", "", "R²")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8, fontsize=8)

    plt.tight_layout()
    out_path = Path(output_dir) / "ablation_study.png"
    plt.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return str(out_path)


# ---------------------------------------------------------------------------
# 4. Feature importance (from attention weights)
# ---------------------------------------------------------------------------

def plot_attention_heatmap(attn_maps: list, feature_names: list,
                            output_dir: str) -> str:
    """Average attention across heads and samples for the last transformer layer."""
    last_layer = attn_maps[-1]                 # (B, n_heads, T, T)
    avg_attn   = last_layer.mean(dim=(0, 1))   # (T, T)
    cls_attn   = avg_attn[0, 1:].numpy()       # CLS→feature attention (skip CLS token)

    n_feat = min(len(feature_names), len(cls_attn))
    feat_labels = feature_names[:n_feat]
    attn_vals   = cls_attn[:n_feat]

    # Sort by importance
    order       = np.argsort(attn_vals)[::-1][:20]  # Top 20
    sorted_lbls = [feat_labels[i] for i in order]
    sorted_vals = attn_vals[order]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.barh(sorted_lbls[::-1], sorted_vals[::-1], color=ACCENT, alpha=0.85)
    _style_ax(ax, "Feature Importance (CLS Attention, Last Layer)",
              "Mean Attention Weight", "Feature")
    plt.tight_layout()

    out_path = Path(output_dir) / "feature_importance_attention.png"
    plt.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return str(out_path)


# ---------------------------------------------------------------------------
# 5. Residual analysis
# ---------------------------------------------------------------------------

def plot_residuals(npz_path: str, output_dir: str) -> str:
    data    = np.load(npz_path)
    preds   = data["preds_orig"]
    targets = data["tgts_orig"]
    names   = ["resilience", "temp", "flood", "energy"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor=DARK_BG)
    fig.suptitle("Residual Analysis — Test Set", color=TEXT, fontsize=13, fontweight="bold")

    for i, name in enumerate(names):
        p, t = preds[:, i], targets[:, i]
        resid = p - t

        # Residual vs predicted
        ax = axes[0, i]
        ax.scatter(p, resid, alpha=0.2, s=4, color=PALETTE[i])
        ax.axhline(0, color=MUTED, lw=1.5, linestyle="--")
        _style_ax(ax, TARGET_DISPLAY[name], "Predicted", "Residual")

        # Residual histogram
        ax = axes[1, i]
        ax.hist(resid, bins=40, color=PALETTE[i], alpha=0.8, edgecolor=DARK_BG)
        ax.axvline(0, color=MUTED, lw=1.5, linestyle="--")
        _style_ax(ax, f"μ={resid.mean():.2f}  σ={resid.std():.2f}",
                  "Residual", "Count")

    plt.tight_layout()
    out_path = Path(output_dir) / "residual_analysis.png"
    plt.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return str(out_path)


# ---------------------------------------------------------------------------
# 6. EDA plots
# ---------------------------------------------------------------------------

def plot_eda(features_csv: str, labels_csv: str, output_dir: str) -> str:
    df_X = pd.read_csv(features_csv)
    df_y = pd.read_csv(labels_csv)
    df   = pd.concat([df_X, df_y[["resilience_score", "temp_reduction_f",
                                   "flood_risk_reduction", "energy_savings_usd"]]], axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor=DARK_BG)
    fig.suptitle("Dataset EDA — Climate Resilience Dataset", color=TEXT,
                 fontsize=13, fontweight="bold")

    targets = ["resilience_score", "temp_reduction_f", "flood_risk_reduction", "energy_savings_usd"]
    labels  = list(TARGET_DISPLAY.values())

    # Target distributions
    for i, (col, lbl) in enumerate(zip(targets, labels)):
        ax = axes[0, i]
        ax.hist(df[col].dropna(), bins=40, color=PALETTE[i], alpha=0.85, edgecolor=DARK_BG)
        _style_ax(ax, lbl, "Value", "Count")

    # Correlation: temp_increase vs resilience by zone
    zones = ["zone_tropical", "zone_subtropical", "zone_temperate", "zone_cold"]
    zone_labels = ["Tropical", "Subtropical", "Temperate", "Cold"]
    zone_colors = [RED, AMBER, GREEN, ACCENT]

    ax = axes[1, 0]
    for zone, zlbl, zcolor in zip(zones, zone_labels, zone_colors):
        mask = df[zone] == 1
        ax.scatter(df.loc[mask, "baseline_temp"],
                   df.loc[mask, "resilience_score"],
                   alpha=0.15, s=4, color=zcolor, label=zlbl)
    _style_ax(ax, "Resilience vs Baseline Temp by Zone",
              "Baseline Temp (°C)", "Resilience Score")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8, fontsize=7)

    # Intervention count vs resilience
    ax = axes[1, 1]
    for n_iv, color in enumerate(PALETTE[:5]):
        mask = df["n_interventions"] == n_iv
        if mask.sum() > 0:
            ax.scatter(df.loc[mask, "year_norm"],
                       df.loc[mask, "resilience_score"],
                       alpha=0.2, s=4, color=color, label=f"{n_iv} interventions")
    _style_ax(ax, "Resilience vs Year by # Interventions",
              "Year (normalised)", "Resilience Score")
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8, fontsize=7)

    # Coastal distance vs flood reduction
    ax = axes[1, 2]
    ax.scatter(df["coastal_distance"].clip(0, 400),
               df["flood_risk_reduction"],
               alpha=0.15, s=4, color=ACCENT)
    _style_ax(ax, "Coastal Distance vs Flood Reduction",
              "Coastal Distance (km)", "Flood Risk Reduction (%)")

    # Solar radiation vs energy savings
    ax = axes[1, 3]
    ax.scatter(df["baseline_solar_radiation"],
               df["energy_savings_usd"].clip(0, 5000),
               alpha=0.15, s=4, color=GREEN)
    _style_ax(ax, "Solar Radiation vs Energy Savings",
              "Solar Radiation (W/m²)", "Energy Savings ($/yr)")

    plt.tight_layout()
    out_path = Path(output_dir) / "eda_plots.png"
    plt.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return str(out_path)
