# 🌍 ClimateProof — AI-Powered Climate Resilience Simulator

> **CMPE258 Final Project — End-to-End MLOps Pipeline**
> Multi-task deep learning for climate adaptation outcome prediction

[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF)](https://github.com/features/actions)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2)](https://mlflow.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C)](https://pytorch.org)

---

## Abstract

ClimateProof trains **ClimateResilienceNet**, a physics-informed transformer model (~520k parameters) that predicts four climate adaptation outcomes simultaneously given location features, IPCC climate projections, and building intervention choices. The end-to-end pipeline covers data generation → training → evaluation → ablation study → hyperparameter sweep → CI/CD → production inference API → interactive web dashboard.

---

## Team Members & Contributions

| Member | Role | Contributions |
|---|---|---|
| **Akshata Madavi** | ML Engineer + Full-Stack Engineer| ClimateResilienceNet architecture, training loop, MLflow/TensorBoard integration, custom loss design, React/Mapbox dashboard |
| **Archana Shivashankar** | Data Engineer + Full-Stack Engineer | Physics-informed dataset generator (IPCC AR6 equations), feature engineering pipeline, RobustScaler preprocessing, FastAPI inference server |
| **Parth Maradia** | Research Lead + Full-Stack Engineer | IPCC AR6 literature review, ablation study design, evaluation metrics framework, model design documentation, GitHub Actions CI/CD pipeline |

---

## What We Built (Beyond Any Existing Code)

Every component was written from scratch for this project:

| Component | Description | Lines |
|---|---|---|
| `ml/models/climate_net.py` | TabTransformer + physics prior + uncertainty estimation | ~370 |
| `ml/data/generate_dataset.py` | IPCC AR6 physics-informed data generator | ~220 |
| `ml/models/losses.py` | ClimateAwareLoss: MSE + monotonicity + NLL + physics constraint | ~120 |
| `ml/training/trainer.py` | Training loop: MLflow + TensorBoard + early stopping + SGDR | ~190 |
| `ml/evaluation/ablation.py` | 8-variant ablation study framework | ~130 |
| `ml/evaluation/visualize.py` | 6 custom dark-theme visualization functions | ~220 |
| `ml/inference/server.py` | FastAPI inference microservice with graceful fallback | ~80 |
| `.github/workflows/train.yml` | 4-job CI/CD: test → train → visualize → deploy | ~160 |
| `climateproof_final.ipynb` | 14-step end-to-end training notebook | ~700 |

---

## Model Architecture — ClimateResilienceNet

```
Input (N × 36 features)
         │
  FeatureTokenizer          — per-feature linear projection → (N, F, d=128)
         │
  PhysicsInformedPrior      — IPCC zone embeddings as learnable residual signal
         │
  4× TransformerEncoder     — pre-LN MHSA (8 heads, d_k=16) + FFN (d_ff=512) + residual
         │
  CLS token + Global Pool   — (N, 128)
         │
  Shared Trunk              — Linear → BatchNorm → SELU → AlphaDropout
         │
  ┌──────┬──────┬──────┬──────┐
  │      │      │      │      │
Resil  Temp  Flood  Energy   — per-task heads, domain-specific activations
(Sig×100)(SP) (Sig×90) (μ+σ²) — Energy head: aleatoric uncertainty estimation
```

**Total trainable parameters: ~520,000**

### Design Choices — Documented

| Parameter | Choice | Rationale |
|---|---|---|
| **Backbone** | TabTransformer (Gorishniy et al. 2021) | Self-attention captures non-linear feature interactions (solar_radiation × green_roof) that gradient boosting misses |
| **Physics prior** | IPCC AR6 zone multiplier embeddings | Warm-starts training from climatically sensible representations; reduces convergence from ~100 to ~60 epochs |
| **Normalisation** | Pre-LayerNorm (Xiong et al. 2020) | Stable gradients without LR warmup — necessary when physics prior and attention co-evolve |
| **FFN activation** | GELU | Smoother gradient than ReLU for approximately Gaussian tabular feature distributions |
| **Head activation** | SELU + AlphaDropout | Self-normalising properties maintain stable gradient norms in deep task heads |
| **Output activation** | Sigmoid×100 (resilience), Sigmoid×90 (flood), Softplus (temp, energy) | Constrain outputs to physical feasibility ranges |
| **Loss** | ClimateAwareLoss | Standard MSE ignores constraints; monotonicity penalty prevents predicting that removing interventions improves resilience |
| **Uncertainty** | Aleatoric NLL on energy head (Kendall & Gal 2017) | Energy savings is log-normally distributed (highest variance); learned σ² gives calibrated uncertainty |
| **Feature scaler** | RobustScaler | Handles outliers in coastal_distance (0–1000km) and building_area (log-normal) |
| **Optimiser** | AdamW (Loshchilov & Hutter 2019) | Decoupled weight decay avoids L2 regularisation conflating with gradient adaptive scaling |
| **Scheduler** | CosineAnnealingWarmRestarts (SGDR) | Escapes local minima from non-convex physics-prior + attention interaction |
| **Data augmentation** | Gaussian noise on continuous features (σ=2%) | Prevents overfitting; zeros noise on binary intervention flags |

---

## Dataset

### Source & Validation

Physics-informed synthetic dataset generated from validated equations:

| Source | Used for |
|---|---|
| IPCC AR6 WG1 (SSP2-4.5) | Temperature trajectory by zone and year |
| EPA Urban Heat Island Program | Green/cool roof temperature reduction |
| DOE Building Energy Databook | Solar + HVAC energy savings by building type |
| FEMA Coastal Flood Hazard Analysis | Flood risk by coastal distance and elevation |
| Santamouris (2014), *Energy & Buildings* | Urban heat island validation |

Gaussian noise (σ = 3% of label value) simulates real-world measurement uncertainty.

### Statistics

| Property | Value |
|---|---|
| Total samples | 50,000 |
| Input features | 36 |
| Output targets | 4 |
| Global cities | 40 |
| Projection years | 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100 |
| Intervention combos | 16 (all 2⁴ subsets of 4 interventions) |
| Building types | 5 (residential, commercial, industrial, mixed_use, institutional) |
| Split | 80% train / 10% val / 10% test (stratified by climate zone) |

---

## Experiments & Results

### Test Set Metrics (ClimateResilienceNet, original scale)

| Output | R² | RMSE | MAE |
|---|---|---|---|
| Resilience Score | 0.941 | 3.8 pts | 2.9 pts |
| Temp Reduction (°F) | 0.921 | 0.42 °F | 0.31 °F |
| Flood Risk Reduction (%) | 0.908 | 3.1 % | 2.3 % |
| Energy Savings ($/yr) | 0.878 | $142 | $98 |
| **Mean** | **0.912** | — | — |

### Ablation Study Results

| Variant | Mean R² | Δ vs Full | Key takeaway |
|---|---|---|---|
| **Full model** | **0.912** | baseline | Best across all outputs |
| No physics prior | 0.873 | −4.3% | Prior provides meaningful inductive bias |
| No attention (MLP) | 0.798 | −12.5% | Self-attention is the most important component |
| Shallow (2 layers) | 0.851 | −6.7% | 4 layers are needed for feature interaction depth |
| Narrow (d=64) | 0.869 | −4.7% | 128-dim embeddings provide meaningful capacity |
| No uncertainty | 0.906 | −0.7% | Minimal R² impact; uncertainty calibration is the gain |
| Random Forest | 0.821 | −10.0% | Strong baseline; DL gains from interaction modeling |
| Linear Regression | 0.612 | −32.9% | Problem is highly non-linear |

### Hyperparameter Sweep (Optuna, 30 trials)

Best trial (TPE sampler): `lr=3.12e-4, d_model=128, n_heads=8, n_layers=4, dropout=0.089`

---

## MLOps Pipeline (Level 2)

```
Data Generation  ──►  Preprocessing  ──►  Training
  50k samples          RobustScaler        AdamW + SGDR
  IPCC AR6 eqns        Log-norm targets    Gradient clipping
  Physics noise        Stratified split    Early stopping (patience=20)
                                                │
                                         MLflow Tracking         TensorBoard
                                         Model Registry          Loss curves
                                         Run comparison          R² per output
                                                │
                                         Evaluation
                                         Ablation Study  ──►  Optuna Sweep
                                                │
                                    GitHub Actions CI/CD
                            ┌────────────────────────────────┐
                            │  Job 1: test (lint + validate)  │
                            │  Job 2: train (quality gate)    │
                            │  Job 3: visualize (plots)       │
                            │  Job 4: deploy (HF Spaces)      │
                            └────────────────────────────────┘
                                                │
                                  FastAPI Inference Server (:8001)
                                    ↕ REST API (JSON)
                                  Express.js Proxy (:5000)
                                    ↕
                                  React + Mapbox Dashboard
```

---

## Inputs & Outputs

### Model Inputs (36 features)

| Group | Features | Count |
|---|---|---|
| Geographic | lat, lon, lat/lon sin/cos, elevation, coastal_distance, log-coastal | 9 |
| Climate Baseline | baseline_temp, humidity, solar_radiation, wind_speed | 4 |
| Climate Zone | zone_tropical/subtropical/temperate/cold (one-hot) | 4 |
| Temporal | year_norm, year_sin, year_cos | 3 |
| Interventions | green_roof, solar_panels, flood_walls, permeable_pavement, n_interventions | 5 |
| Building | building_age, building_area, log-area, building_type | 4 |
| Interactions | solar×green, solar×solar_panel, coast×flood_wall | 3 |

### Model Outputs (4 targets)

| Output | Range | Unit | Uncertainty? |
|---|---|---|---|
| `resilience_score` | 0–100 | points | No |
| `temp_reduction_f` | 0–15 | °F | No |
| `flood_risk_reduction` | 0–90 | % | No |
| `energy_savings_usd` | 0–5000+ | $/yr | **Yes** (aleatoric σ) |

---

## Quick Start

```bash
# 1. Clone & install Python deps
git clone https://github.com/YOUR_ORG/Climate-Resilience-Simulator.git
cd Climate-Resilience-Simulator
pip install -r requirements.txt

# 2. Generate dataset + train model
python -m ml.training.train

# 3. Monitor training
tensorboard --logdir ml/artifacts/tensorboard
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 4. Start inference server
uvicorn ml.inference.server:app --port 8001

# 5. Start web dashboard
npm install && npm run dev
# → http://localhost:5000

# 6. Run interactive notebook (Gradio + full pipeline)
jupyter notebook climateproof_final.ipynb
```

### CLI Options

```bash
# Override config
python -m ml.training.train --epochs 150 --lr 1e-3 --d-model 256

# Disable physics prior (ablation)
python -m ml.training.train --no-physics-prior

# Hyperparameter sweep (30 Optuna trials)
python -m ml.training.train --sweep --n-trials 30
```

---

## Repository Structure

```
Climate-Resilience-Simulator/
├── ml/                              # ML package
│   ├── data/
│   │   ├── generate_dataset.py      # Physics-informed 50k generator (IPCC AR6)
│   │   ├── dataset.py               # PyTorch Dataset + DataLoaders
│   │   └── preprocessor.py          # Feature engineering + RobustScaler
│   ├── models/
│   │   ├── climate_net.py           # ClimateResilienceNet (~520k params)
│   │   ├── baselines.py             # Linear / RF / GBT / MLP baselines
│   │   └── losses.py                # ClimateAwareLoss + HuberMultiTaskLoss
│   ├── training/
│   │   ├── trainer.py               # MLflow + TensorBoard training loop
│   │   └── train.py                 # CLI entry point + Optuna sweep
│   ├── evaluation/
│   │   ├── metrics.py               # R², RMSE, MAE
│   │   ├── evaluate.py              # Test set evaluation
│   │   ├── ablation.py              # 8-variant ablation runner
│   │   └── visualize.py             # 6 dark-theme visualization functions
│   ├── inference/
│   │   ├── predictor.py             # End-to-end inference pipeline
│   │   └── server.py                # FastAPI microservice
│   └── configs/
│       └── config.yaml              # Hyperparameters (annotated)
├── client/                          # React + TypeScript frontend
│   └── src/pages/dashboard.tsx      # Interactive map dashboard + ML panel
├── server/                          # Express.js backend
│   └── routes.ts                    # API proxy + /api/ml-predict endpoint
├── .github/workflows/train.yml      # CI/CD: test → train → viz → deploy
├── climateproof_final.ipynb         # 14-step end-to-end notebook
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Related Work

- Gorishniy et al. (2021) — *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer baseline)
- Xiong et al. (2020) — *On Layer Normalization in the Transformer Architecture* (Pre-LN choice)
- Kendall & Gal (2017) — *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* (aleatoric NLL)
- Loshchilov & Hutter (2019) — *Decoupled Weight Decay Regularization* (AdamW)
- IPCC AR6 Working Group 1 (2021) — Climate projections, SSP2-4.5 scenario
- Santamouris (2014) — *Cooling the cities — a review of reflective and green roof technologies*

---

*ClimateProof Team | CMPE258 Final Project | San Jose State University*
