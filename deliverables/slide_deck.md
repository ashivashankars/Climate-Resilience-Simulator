---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #121212
color: #e0e0e0
---

# 🌍 ClimateProof: AI-Powered Climate Resilience Simulator
**CMPE258 Final Project — End-to-End MLOps Pipeline**

*Akshata Madavi, Archana Shivashankar, Parth Maradia*
San Jose State University

---

# Abstract & Problem Statement

* **Problem:** Predicting urban resilience against climate change requires complex, nonlinear models incorporating various climate policies and building adaptations. Existing simulations are often manual and disconnected from interactive ML tools.
* **Our Solution:** **ClimateResilienceNet**, a physics-informed TabTransformer model that predicts 4 climate adaptation outcomes simultaneously: Resilience Score, Temperature Reduction, Flood Risk Reduction, and Energy Savings.
* **Core Technology:** Deep Learning (TabTransformer), Multi-Task Learning, MLOps Level 2 (Automated CI/CD).

---

# What We Built

* **Custom ML Architecture**: Built from scratch using PyTorch (520k params), leveraging a physics-informed prior and TabTransformer backbone.
* **Synthetic Data Generation**: Physics-informed 50k samples generator using IPCC AR6 WG1 equations.
* **Complex Loss Function**: `ClimateAwareLoss` handling MSE, monotonicity penalty, and aleatoric uncertainty on energy outputs.
* **Full MLOps Pipeline**: GitHub Actions for CI/CD, MLflow for experiment tracking, TensorBoard for visualization.
* **Full-Stack Application**: FastAPI inference server proxying to an Express/React dynamic interactive dashboard.

---

# Model Architecture: ClimateResilienceNet

* **Inputs**: 36 features (Geographic, Climate Baseline, Climate Zone, Temporal, Interventions, Building characteristics).
* **Feature Representation**: `FeatureTokenizer` + IPCC Zone embeddings as a learnable residual signal.
* **Attention Mechanism**: 4x Transformer Encoders with Pre-LN MHSA.
* **Multi-Task Output Heads**: 4 specific heads for Resilience, Temp, Flood, Energy. Uses custom activations (`Sigmoid×100`, `Sigmoid×90`, `Softplus`).

---

# Methodology & Experiments

* **Metrics Evaluated**: R², RMSE, MAE.
* **Results**: Reached Mean R² of **0.912** on the Test Set. Resilience Score R² = 0.941.
* **Ablation Studies**:
    * Removing self-attention dropped performance by 12.5%.
    * Removing physics prior dropped performance by 4.3%.
    * Deep learning provided significant gains over Random Forest & Linear Regression baselines.
* **Hyperparameter Sweeps**: Ran 30 Optuna trials for automated tuning.

---

# MLOps Architecture (Maturity Level 2)

* **Data Pipeline**: RobustScaler and stratified splitting.
* **Training & Tracking**: `AdamW` + `SGDR` schedulers, logging to MLflow and TensorBoard.
* **CI/CD Automation**: GitHub Actions triggers 4 jobs on push:
    1. Test (Lint + Validate)
    2. Train (Quality Gate)
    3. Visualize (Plots)
    4. Deploy (Mockup target: HF Spaces)
* **Model Inference**: Production-ready FastAPI microservice with graceful "rule-based fallback" degradation.

---

# Team Contributions

* **Akshata Madavi**: ML Engineer + Full-Stack. (ClimateResilienceNet architecture, training loop, MLflow, React Dashboard)
* **Archana Shivashankar**: Data Engineer + Full-Stack. (IPCC AR6 physics data generator, Feature engineering, FastAPI)
* **Parth Maradia**: Research Lead + Full-Stack. (Literature review, Ablation framework, Evaluation metrics, CI/CD pipeline)

---

# Conclusion & Future Work

* **Key Takeaway**: Physics-informed inductive biases coupled with TabTransformer attention mechanisms significantly outperform standard regressors on multivariate climate data.
* **Future Work**:
    * Integrate real-world telemetry data via IoT.
    * Expand interventions payload to capture more nuanced economic factors.
    * Elevate to MLOps Level 4 (Auto-retraining and drift detection).

---
# Q&A

Thank you! Let's explore the demo.
