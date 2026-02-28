---
title: Network Intrusion Detection
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

# Network Intrusion Detection System (End-to-End ML + MLOps Project)

An end-to-end Machine Learning pipeline for detecting network intrusions using real-world MLOps techniques. This project goes beyond model training — it includes a complete training pipeline, modular architecture, data validation with drift detection, MLflow integration via DagsHub, and a FastAPI-based deployment layer.

## Key Features
- Statistical model selection: Bootstrap (n=100) + McNemar's test (p<0.05)
- Champion-Challenger framework — promotes only on statistically significant gains
- SHAP explainability — global feature importance + individual waterfall plots
- KS-Test drift detection across all features
- MLflow + DagsHub experiment tracking
- FastAPI for serving model predictions
- Streamlit monitoring dashboard

## Tech Stack
- **ML**: Scikit-learn (Random Forest, Gradient Boosting, AdaBoost, LR, DT)
- **Statistical Rigor**: SciPy (McNemar's test, KS-test, Bootstrap CI)
- **Experiment Tracking**: MLflow + DagsHub
- **Serving**: FastAPI + Uvicorn
- **Monitoring**: Streamlit + SHAP
