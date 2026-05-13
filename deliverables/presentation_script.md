# ClimateProof: Presentation Script

## Slide 1: Title Slide
**Parth:** "Hello everyone! We are excited to present our CMPE258 final project, ClimateProof. We are team Archs: Akshata, Archana, and myself, Parth. Our project is an end-to-end MLOps pipeline and application focused on an AI-powered Climate Resilience Simulator."

## Slide 2: Abstract & Problem Statement
**Parth:** "The impact of climate change on urban environments is complex. City planners and engineers struggle to predict how combining different building adaptations—like green roofs or solar panels—will perform under various IPCC climate projections over decades. Existing simulations are highly manual, siloed, and computationally heavy. Our solution is **ClimateResilienceNet**, a multi-task deep learning model capable of simultaneously predicting resilience scores, temperature reduction, flood risk, and energy savings. We’ve wrapped this in a full MLOps pipeline and an interactive dashboard."

## Slide 3: What We Built
**Parth:** "We built every component of this project from scratch, far beyond what exists in basic tutorials. We implemented a custom TabTransformer architecture in PyTorch, consisting of around 520,000 parameters. Because real-world high-quality granular data is scarce, we engineered a physics-informed dataset generator using IPCC AR6 WG1 equations to produce 50,000 highly realistic samples. We designed a `ClimateAwareLoss` function, an MLOps Level 2 automated CI/CD pipeline via GitHub Actions, and an Express/React-based full-stack interactive dashboard for our inference service."

## Slide 4: Model Architecture
**Akshata:** "Thank you, Parth. For the model architecture, I designed **ClimateResilienceNet**. It takes 36 features representing geography, baseline climate, temporal features, and intervention choices. We process this through a `FeatureTokenizer`, applying an IPCC Zone embedding as a warm-starting physics prior. The core backbone is a 4-layer Transformer Encoder with pre-LayerNorm self-attention, which efficiently captures non-linear feature interactions—like the interaction between solar radiation and green roofs. Finally, the shared trunk branches into four task-specific output heads with custom activations like `Sigmoid×100` for physical boundary constraints."

## Slide 5: Methodology & Experiments
**Akshata:** "Our model achieved an impressive Mean R² of 0.912 on the test set, with our resilience score prediction hitting an R² of 0.941. We conducted extensive ablation studies. We found that removing self-attention caused a 12.5% drop in performance, proving that interaction modeling was key. Similarly, the physics prior provided a 4.3% boost and reduced our convergence time dramatically. Deep learning massively outperformed baseline linear and random forest regressors on this highly nonlinear problem. We also tuned everything via a 30-trial Optuna hyperparameter sweep."

## Slide 6: MLOps Architecture
**Archana:** "Moving to production, I'd like to highlight our MLOps Level 2 infrastructure. Our data pipeline automatically handles RobustScaling and stratified splitting. During training, we use `AdamW` optimizers and cosine annealing schedulers, with all artifacts perfectly tracked in MLflow and TensorBoard. Our GitHub Actions CI/CD automatically runs linting, validation, trains a quality-gate model, generates visualizations, and pushes to our deployment targets. The FastAPI inference server uses graceful degradation—if the ML model goes offline, the Node backend seamlessly switches to a rule-based physics fallback so the UI never crashes."

## Slide 7: Team Contributions
**Archana:** "This was a highly collaborative effort. Akshata handled the core ML architecture, custom loss design, MLflow integration, and React dashboard. I focused on the data engineering, the physics-informed IPCC data generator, and the FastAPI inference server. Parth led the research literature review, orchestrated the ablation studies and evaluation metrics, and architected our robust GitHub Actions CI/CD pipelines."

## Slide 8: Conclusion & Q&A
**Parth:** "In conclusion, we successfully demonstrated that embedding physics priors into deep self-attention architectures drastically improves multi-task climate predictions. In the future, we plan to ingest live IoT telemetry and scale to an MLOps Level 4 architecture. We invite you to check out our fully documented repository and interactive dashboard. Thank you! We’ll now take questions."
